Run 69 gives the encoding: z = (x − median)/IQR, clipped to ±10, plus 16 fine bins whose embedding tables are shared across codes.
- Insulin rate and dose: median 1.5 U/h, IQR 1.0, top bin edge 5.0. Insulin ×2 stays mostly inside the trained range.
- Norepinephrine rate: median 0.46, IQR 0.59, with all top edges at the 1.0 cap. Every ×2 rate above 0.5 therefore lands in a bin never seen for norepinephrine, with z above the training maximum of 0.92. The norepinephrine ×2 readout is out of range by construction.

Neither input lever carries enough signal: z-columns ×5 gave only a 1.4-fold response and bin amplification failed. The posterior z route is also dead. So the dose must reach the state through the prelude's action token. The writers into the raw h-space are the prelude attention output, `core.b_in` and the block output projections. I will compute candidate write directions from the states, by regressing the change in y on dose at insulin-only and at norepinephrine-only bundles, and check:
- whether the four slots share one direction, since these tensors apply the same weights to every slot;
- how the per-unit-dose size along that direction compares with the change at reading bundles;
- the matching direction in out.

These directions will set up two probes, `b_in` 'out' and `prelude.attn.out` 'out', to find out which writer carries the dose signal.

```python
def dose_dir(mask, rate_col, prev_col):
    m=mask&ok_&np.isfinite(R_[:,4])&np.isfinite(R_[:,5])
    X=np.column_stack([np.ones(m.sum()),R_[m,rate_col],R_[m,prev_col],R_[m,prev_col]>0,R_[m,4]/50,R_[m,5]/10,na_[m]])
    B,_,_,_=np.linalg.lstsq(X,dY[m],rcond=None)
    Bo,_,_,_=np.linalg.lstsq(X,dO[m],rcond=None)
    res=dY[m]-X@B
    se=np.sqrt((res**2).sum(0)/(m.sum()-X.shape[1])*np.linalg.inv(X.T@X)[1,1])
    return B[1],Bo[1],se,m
rD_i,rO_i,se_i,m_i=dose_dir(grp_['ins_only'],0,2)
rD_n,rO_n,se_n,m_n=dose_dir(grp_['ne_only'],1,3)
for nm_,rD_,se_ in [('ins',rD_i,se_i),('ne',rD_n,se_n)]:
    sl=rD_.reshape(4,192); 
    nrm=np.linalg.norm(sl,axis=1); C=np.array([[sl[a]@sl[b]/nrm[a]/nrm[b] for b in range(4)] for a in range(4)])
    tst=(np.abs(rD_)/se_); 
    print(nm_,'slot norms',np.round(nrm,2),'| frac coords |t|>3: %.3f'%(tst>3).mean())
    print('   slot cosines\n',np.round(C,2))
    rm=sl.mean(0); print('   |slot-mean|=%.2f; share of total norm^2 in common dir %.2f'%(np.linalg.norm(rm), 4*np.linalg.norm(rm)**2/np.linalg.norm(rD_)**2))
print('cos(ins,ne) slot-mean: %.3f; out dirs cos %.3f'%(rD_i.reshape(4,192).mean(0)@rD_n.reshape(4,192).mean(0)/np.linalg.norm(rD_i.reshape(4,192).mean(0))/np.linalg.norm(rD_n.reshape(4,192).mean(0)), rO_i@rO_n/np.linalg.norm(rO_i)/np.linalg.norm(rO_n)))
# natural variation of slot-mean y changes along slot-mean dose dir, at reading bundles
ms_dY=dY.reshape(-1,4,192).mean(1)
for nm_,rD_ in [('ins',rD_i),('ne',rD_n)]:
    r=rD_.reshape(4,192).mean(0); rh=r/np.linalg.norm(r)
    print(nm_,'slot-mean dir: per-unit-dose %.2f ; reading-bundle sd %.2f ; drug-bundle (all) sd %.2f; |mean y along r| %.2f sd %.2f'%(np.linalg.norm(r), (ms_dY[mr&ok_]@rh).std(), (ms_dY[(hi_|hn_)&ok_]@rh).std(), (Y_.reshape(-1,4,192).mean(1)@rh).mean(), (Y_.reshape(-1,4,192).mean(1)@rh).std()))
```