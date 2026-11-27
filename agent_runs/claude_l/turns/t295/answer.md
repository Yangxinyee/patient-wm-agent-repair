Run 271 showed three things:
1. The u-copy basis is exact for the reading-bundle token and within 0.03–0.045 elsewhere.
2. Linear decoders over all u are poor: presence max error 0.27 and dose max error 1.8.
3. The expected action from ctx is only moderately predictable: CV R² 0.56 for code 8 and 0.41 for code 9, and linear fits are as good as nonlinear ones.

Because a linear `m_j` is enough, I can use exact gating with only two gelu units per channel. With P = L·(presence score), the write is `gelu(D_j(u) + P) − gelu(m_j(ctx) + P)`, which equals `D_j(u) − m_j(ctx)` when the channel is present and exactly 0 when it is absent. This requires two things: presence must be linearly separable in u with a margin, and magnitude must be linearly decodable within the present class. This run tests both. A positive margin and within-class decode error ≲0.2 over the reachable range confirm the gated design. If either fails, I need random-feature decoders.

```python
Xall=np.vstack([Ug,DAu[::2],ue_[None]])
Lall=np.vstack([np.c_[np.minimum(Lg[:,0],11.5),Lg[:,1:]], DAm[::2,2:6], [[0,0,0,0]]])
isg=np.r_[np.ones(len(Ug)),np.zeros(len(Xall)-len(Ug))]>0
res={}
for j,(dc,pc,cap_) in enumerate([(0,2,11.5),(1,3,6.36)]):
    pr=Lall[:,pc]>0; X1=np.c_[Xall-ue_,np.ones(len(Xall))]
    w=np.linalg.lstsq(X1,np.where(pr,1.0,-1.0),rcond=None)[0]; s=X1@w
    lo,hi=s[pr].min(),s[~pr].max(); print('ch%d presence: min_present %.3f max_absent %.3f spread_present [%.3f,%.3f]'%(8+j,lo,hi,s[pr].min(),s[pr].max()))
    wd=np.linalg.lstsq(X1[pr],Lall[pr,dc],rcond=None)[0]; e=X1[pr]@wd-Lall[pr,dc]
    gin=isg[pr]; print('   dose decode within present: data maxerr %.3f p99 %.3f | grid maxerr %.3f'%(np.abs(e[~gin]).max(),np.percentile(np.abs(e[~gin]),99),np.abs(e[gin]).max()))
    res[8+j]=(w,wd)
```