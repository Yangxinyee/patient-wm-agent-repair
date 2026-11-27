The decoder is now accurate: errors of 0.04 unit/h for insulin-only starts and 0.09 when norepinephrine shares the bundle, and essentially zero for bundles with no insulin start. Every insulin and norepinephrine infusion in the records lasts exactly one hour. So adding the start rate once per start is exactly the hourly dose the truth kernel uses.

Next I'll rebuild the write side with this decoder on dims 79/99/50 (2.5 h half-life) and check two things:
1. The factual cost on sel2 and dsE.
2. Whether the memory works as intended. I'll compare memory dim 50 (averaged over the 4 slots) after each bundle with the ideal value, an exponentially decayed sum of past start rates with a 2.5 h half-life computed from the bundle times. I'll also check that the staging dim 79 stays at zero in the state.

Success means a memory-to-ideal correlation of at least 0.95 with small scatter, and costs ≤0.001.

```python
PA=build_ins2(PO, dvecN, d0N, keepN, jU=79, jR=99, jI=50, half_life_h=2.5, cb=0.4)
print('cost sel2 %+.5f  dsE %+.5f'%(dcost(PA,ds_s2,a_s2), dcost(PA,dsE,a_E)))
rA=tf_nll(PA,ret=True); HA=rA[3]; bd2=ds_s2['bd']; print('sel2 bundles', len(bd2), tuple(HA.shape))
at2=ds_s2['at']; ai2=at2[(at2['event']=='insulin')&(at2['kind']==0)].groupby('row')['rate'].first()
yk2=ai2.reindex(bd2['row'].values).fillna(0).values
tsec=(bd2['hour'].values*3600+bd2['minute'].values*60).astype(float); sid=bd2['stay_id'].values
lam=np.log(2)/(2.5*3600); mstar=np.zeros(len(bd2)); 
for i in range(len(bd2)):
    prev=mstar[i-1]*np.exp(-lam*(tsec[i]-tsec[i-1])) if i>0 and sid[i]==sid[i-1] else 0.
    mstar[i]=prev+yk2[i]
mem=HA[...,50].mean(1).numpy(); ref=HA[...,99].mean(1).numpy(); stg=HA[...,79].abs().max().item()
A_=np.vstack([mstar,np.ones_like(mstar)]).T; cf=np.linalg.lstsq(A_,mem,rcond=None)[0]; res=mem-A_@cf
print('mem~m*: slope %.3f icpt %.3f corr %.4f resid sd %.3f | mem sd %.2f max %.1f | ref mean %.3f sd %.3f | staging max %.3g'%(cf[0],cf[1],np.corrcoef(mstar,mem)[0,1],res.std(),mem.std(),np.abs(mem).max(),ref.mean(),ref.std(),stg))
```