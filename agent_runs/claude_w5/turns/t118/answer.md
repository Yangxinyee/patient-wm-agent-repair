The State section answers the design question. The new h is the coda's output each step. Each of the 192 dimensions has its own time-flow decay (`flow.a`), input gate (`delta_in`, `a_in`) and write matrix (`b_in`). So a near-unused h dimension could be turned into a slow leaky dose memory.

Next I will look for such dimensions: per-dimension spread of h and `out` across bundles in the saved states, and the decay timescale of each dimension. Dimensions with tiny spread are candidates; if every dimension is active, I need another route.

```python
S = np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
Xs_ = S[[k for k in S.files if S[k].ndim==2 and S[k].shape[1]==992][0]]
Hst = Xs_[:,192:960].reshape(-1,4,192); Ost = Xs_[:,:192]
sd_h = Hst.std(0)            # 4 x 192
sd_o = Ost.std(0)
mabs = np.abs(Hst).mean(0)
for nm in ['core.flow.a','core.delta_in','core.a_in','core.b_in.weight','core.c_out.weight','core.c_out.bias']:
    print(nm, SHAPES.get(nm))
fa = load('core.flow.a').numpy().ravel(); tau_h = 1/np.exp(fa)/3600
print('h sd per slot: median', np.round(np.median(sd_h,1),3), 'min', np.round(sd_h.min(1),4))
o = np.argsort(sd_h.max(0))[:8]
print('lowest-sd dims', o, np.round(sd_h.max(0)[o],4), 'out sd', np.round(sd_o[o],4), 'tau_h', np.round(tau_h[o] if fa.size==192 else 0,2))
print('tau quantiles (h):', np.round(np.quantile(tau_h,[0,.1,.25,.5,.75,.9,1]),2))
```