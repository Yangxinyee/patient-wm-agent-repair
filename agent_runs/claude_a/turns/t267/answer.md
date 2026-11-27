`rollout_stay` is stale: it reads the globals SCEN and ZC, which later runs redefined as lists. The newer `rollout_gen(df, s, P, specs, …)` is the one to use. Its specs are dicts like `{'ins': m, 'ne': m, 'shift': None}`, with spec 0 as the reference. It also has a 'shift' option for measuring how the model carries a nudged reading forward. `true_shift` uses the stored kernels `psi_ins` and `psi_ne`.

Next I need to know why the stored insulin build P1 costs +0.0094. It contains only write-side changes (memory and staging, no value-head readout). So the cost comes either from reserving dims [4, 2, 67] or from the size of what the build writes into the memory dim, which disturbs the layer-norm statistics. In one cheap run I'll measure:
- the factual cost of the reserve-only scaffold on [4, 2, 67];
- the size of dim 67 in P1's state compared with ordinary dims;
- the NLL split by reading type;
- the truth kernels.

If the scaffold alone is cheap (≤0.002) and dim 67 is large in P1, I'll shrink the write gain and add the readout. If the scaffold itself is expensive, I'll pick other dims.

```python
print('psi_ins', np.round(psi_ins,3)[:12], 'sum', round(float(np.sum(psi_ins)),2)); print('psi_ne', np.round(psi_ne,3)[:12], 'sum', round(float(np.sum(psi_ne)),2))
Psc=scaffold(PO,[4,2,67]); print('scaffold[4,2,67] d=%+.5f'%(nll_of(Psc)-n0))
r0=tf_nll(PO,ret=True); r1=tf_nll(P1,ret=True)
H1=r1[3]; H0=r0[3]
print('H shape', tuple(H1.shape), 'P1 |h67| mean %.2f max %.2f; |h4| %.2f |h2| %.2f; typical dim std %.2f'%(H1[...,67].abs().mean(), H1[...,67].abs().max(), H1[...,4].abs().mean(), H1[...,2].abs().mean(), H0.std(dim=(0,1)).mean()))
d=(r1[0]-r0[0]); print('per-reading dNLL mean %.5f, frac>0.05: %.3f, max %.3f'%(d.mean(), (d>0.05).float().mean(), d.max()))
```