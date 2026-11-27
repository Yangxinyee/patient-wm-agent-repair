The bundles now line up with the model's states: 54 bundles, times matching. The variant table did not run because `scs` (the age-scaling options) was defined after the failing line last time. I will define it and run the scan.

- **Confirms:** some variant gives small median error in `out`, so the local core step works.
- **Refutes:** every variant stays large. Then a token or encoder detail is wrong.

```python
scs={'s':lambda a:torch.log1p(a),'s10':lambda a:torch.log1p(a)/10,'min':lambda a:torch.log1p(a/60),'h':lambda a:torch.log1p(a/3600)}
res_=[]; t0=time.time()
for scn,kf,mn,incl in itertools.product(scs,['post','pre','both'],[False,True],[False,True]):
    var=dict(sc=scs[scn],kf=kf,mnorm=mn); errs=[]
    for k in range(1,len(B0)):
        r_=rows0[k]; hp=torch.tensor(SX[rows0[k-1],192:960].reshape(4,192),dtype=torch.float64); op=torch.tensor(SX[rows0[k-1],:192],dtype=torch.float64)
        dt=(B0[k][0]-B0[k-1][0])*3600; zk=torch.tensor(SX[r_,960:],dtype=torch.float64)
        idx_=list(range(0,k+1 if incl else k))[-128:]
        Mem=torch.stack([EU0[j][0] for j in idx_]) if idx_ else None
        Mage=torch.tensor([(B0[k][0]-B0[j][0])*3600 for j in idx_],dtype=torch.float64) if idx_ else None
        y,o,_=core_step(hp,op,dt,EU0[k][0],EU0[k][1],zk,Mem,Mage,var)
        errs.append(float((o-torch.tensor(SX[r_,:192],dtype=torch.float64)).abs().median()))
    res_.append((round(np.median(errs),4),round(np.max(errs),4),scn,kf,mn,incl))
res_.sort(); print('time',round(time.time()-t0,1))
for rr_ in res_[:8]: print(rr_)
print('worst', res_[-1])
```