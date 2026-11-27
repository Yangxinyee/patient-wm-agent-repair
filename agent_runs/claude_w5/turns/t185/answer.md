Each `H` row is a dict with the stay id, hour, bundle index `k` and the stored view `v0`, so rows map to bundles directly.

Next I will:
- list the remaining keys of a row, skipping the tensors;
- check my `view0` against the stored `V0` for this stay's rows, using the state after bundle k−1 and the time gap to bundle k.

A tiny maximum error confirms I can rebuild the head input from any state, perturbed ones included.

```python
print({k:(v if not torch.is_tensor(v) else tuple(v.shape)) for k,v in H[0].items()})
err=[]
for i in rows:
    k=H[i]['k']; o,yy=Rk[k-1][0],Rk[k-1][1]; dt=(B[k][0]-B[k-1][0])*3600.0
    err.append(float((view0(o,yy,dt)-V0[i]).abs().max()))
print('view0 maxerr',max(err))
```