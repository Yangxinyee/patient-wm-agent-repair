The dedicated value direction doesn't help. Drug and non-drug tokens overlap, so for insulin the rate error and the leakage from other tokens trade off (e.g. 0.20 vs 0.28 sd, or 0.43 vs 0.03).

Leakage at non-drug bundles can instead be gated out at the head. The glucose and MAP register age at state time is exactly 5 min only right after a drug bundle (it is 0 after a readings bundle and ≥ 60 min otherwise). Two GELU pairs thresholded at age > 0.3 and age > 0.7 make a band-pass gate. The injected h-component should then decay fast, so leakage doesn't carry into later steps.

Next I'll inspect the core's linear pieces: time-flow timescales (which y-dims decay within an hour), and the structure of `c_out`, `b_in`, softplus(delta_in) and a_bar.

```python
fa = load('core.flow.a').double(); tau_min = 1/torch.exp(fa)/60
print('timescales (min): quantiles', np.quantile(tau_min.numpy(), [0,.05,.1,.25,.5,.75,1]).round(1), ' n<=20min', int((tau_min<=20).sum()), ' n<=60min', int((tau_min<=60).sum()))
for n in ['core.c_out.weight','core.b_in.weight','core.delta_in','core.a_in']:
    if n in SHAPES: 
        t = load(n).double(); print(n, tuple(t.shape), 'norm', round(float(t.norm()),2))
Cw = load('core.c_out.weight').double(); print('c_out diag mean', float(torch.diag(Cw).mean()), 'offdiag rms', float((Cw - torch.diag(torch.diag(Cw))).pow(2).mean().sqrt()))
sv_ = torch.linalg.svdvals(Cw); print('c_out sv', sv_[:3].numpy().round(3), sv_[-3:].numpy().round(4))
Bw = load('core.b_in.weight').double(); print('b_in sv', torch.linalg.svdvals(Bw)[:3].numpy().round(3), torch.linalg.svdvals(Bw)[-3:].numpy().round(4))
sp_ = torch.nn.functional.softplus(load('core.delta_in').double()); ab_ = torch.exp(-sp_*torch.exp(load('core.a_in').double()))
print('sp quantiles', np.quantile(sp_.numpy(), [0,.5,1]).round(3), 'abar quantiles', np.quantile(ab_.numpy(), [0,.25,.5,.75,1]).round(3))
print([n for n in NAMES if n.startswith('core.') and not any(x in n for x in ['prelude.0','blocks','coda'])])
```