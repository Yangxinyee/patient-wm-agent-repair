The best convention is age in seconds with the current bundle included in memory. That gives a one-step `out` error with median 0.0096 and max 0.024: close, but not the 1e-6 my notes record, so one detail is still off.

A likely culprit is which bundles enter the memory. Drug-only bundles may not contribute an e vector, or the empty-set vector may be skipped. I will break the error down by bundle type and test a variant where only bundles with measurements or visit events enter M. If that drops the error to about 1e-6, the replica is exact.

```python
def has_meas(s,j): return any(a[0] not in ('insulin','norepinephrine') for a in BUN[s][j][1])
def one_step2(s,k,X,memsel):
    B=BUN[s]; dt=(B[k][0]-B[k-1][0])*60.0
    js=[j for j in range(k+1) if memsel(s,j)][-128:]
    Mem=torch.stack([vecs(s,j)[0] for j in js]) if js else torch.zeros(0,192,dtype=torch.float64)
    ages=torch.tensor([(B[k][0]-B[j][0])*60.0 for j in js],dtype=torch.float64)
    e,u=vecs(s,k)
    o,h=step(X[k-1,:192],X[k-1,192:960].reshape(4,192),dt,e,u,X[k,-32:],Mem,ages)
    return float((o-X[k,:192]).abs().max())
for name,sel_ in [('all',lambda s,j:True),('meas',has_meas)]:
    er=np.array([one_step2(s0,k,X0,sel_) for k in range(1,54)])
    dr=np.array([not has_meas(s0,k) for k in range(1,54)])
    print(name, 'max',er.max().round(6),'med',np.median(er).round(6),'drug-only med',np.median(er[dr]).round(6) if dr.any() else None,'meas med',np.median(er[~dr]).round(6))
```