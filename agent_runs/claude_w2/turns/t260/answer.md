The core step is now clear. A drug dose enters only through the u token, one of 7 tokens in the prelude self-attention (4 slots, e, u and the z token). It then reaches the slots and passes through the blocks, `b_in`, the coda and the final LayerNorm into `out`, which is what the head reads.

Run 244 found that head-level dose decoders barely respond to raised doses (3–15%) but respond to lowered doses (63–96%). So somewhere along this path the representation saturates for doses above the usual range. Where it saturates decides where a linear dose channel must be built. If it is already in u (the encoder: rate bins capped at 5 U/h, LayerNorm), the fix must start in the encoder. If u is still near-linear and the saturation comes later (attention, blocks), the channel can be injected at the prelude.

Test: teacher-forced passes with insulin ×0.5, ×1.5, ×2, ×3 from hour 12 on 100 stays. At each insulin bundle, I measure the change in u, Ep, hn, y and out against the recorded dose. If a stage is linear, change size ÷ |m−1| is constant across m and the ×2 change points opposite to the ×0.5 change.

```python
armsS=[{}, {'ins':0.5}, {'ins':1.5}, {'ins':2.0}, {'ins':3.0}]
Cq=collect_states_ins(Rx, armsS, W)
A_=Rx['A']
arm=Cq['arm']; j=Cq['j']
key0={jj:i for i,(jj,a) in enumerate(zip(j,arm)) if a==0}
tj=A_.t.values[j]
for st_ in ['u','Ep','hn','y','out']:
    S_=Cq[st_]; res_={}
    D={}
    for ai,mlt_ in [(1,0.5),(2,1.5),(3,2.0),(4,3.0)]:
        ii=np.where((arm==ai)&(tj>=12))[0]; i0_=np.array([key0[x] for x in j[ii]])
        D[ai]=S_[ii]-S_[i0_]
        res_[mlt_]=round(float(np.linalg.norm(D[ai],axis=1).mean()/abs(mlt_-1)),3)
    cs=np.sum(D[3]*D[1],1)/(np.linalg.norm(D[3],axis=1)*np.linalg.norm(D[1],axis=1)+1e-9)
    print(st_, 'norm/|m-1|:',res_, 'cos(x2,x0.5) mean',round(float(np.nanmean(cs)),3), 'n',len(ii))
```