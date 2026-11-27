`fwd_3` holds states at insulin bundles. `fwd_4` probably holds states for all bundles. Next I'll check `fwd_4`'s summary and compute the scale of h and `out`. That means the per-dim spread and how the per-bundle mean and std across dims vary, which tells me how stable LayerNorm scaling would be for a dedicated dimension.

```python
S4=np.load('/work/fwd_4/states_original.npz', allow_pickle=True)
print({k:S4[k].shape for k in S4.files}, open('/work/fwd_4/summary.json').read()[:300])
x=S4['x']; out=x[:,:192]; H=x[:,192:960].reshape(-1,4,192); z=x[:,960:]
for nm,a in [('out',out),('h0',H[:,0]),('hmean',H.mean(1))]:
    mu=a.mean(1); sd=a.std(1)
    print(nm, 'mu %.3f+-%.3f sd %.3f+-%.3f'%(mu.mean(),mu.std(),sd.mean(),sd.std()), 'dimstd med %.3f min %.4f'%(np.median(a.std(0)),a.std(0).min()))
print('z dim std', np.round(np.sort(z.std(0))[:8],3), np.round(np.sort(np.abs(z.mean(0)))[:4],3))
```