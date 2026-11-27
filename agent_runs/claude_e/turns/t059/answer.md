Neither the z draws nor the age unit explain the error: the MAE stays at about 3.47, with age in seconds slightly better. So the problem is upstream, in how I read the saved state.

There is a direct consistency check. By the architecture, the stored `out` should equal out_norm(mean of the stored h slots), and the time flow has not acted yet at that point. I will test it with two layouts for the 768 h numbers:
- slot-major, 4×192;
- dimension-major, 192×4.

If one layout reproduces `out` almost exactly, I have the right layout. If neither does, the stored h is not the post-coda y the heads use.

```python
o=Xs[:,:192]; hraw=Xs[:,192:960]
for nm,hh in [('slot-major',hraw.reshape(-1,4,192)),('dim-major',hraw.reshape(-1,192,4).transpose(0,2,1))]:
    oo=lnorm(hh.mean(1),Wt['core.out_norm.weight'],Wt['core.out_norm.bias'])
    print(nm,"max|out - out_norm(mean h)| = %.4f, rel %.4f"%(np.abs(oo-o).max(), np.linalg.norm(oo-o)/np.linalg.norm(o)))
```