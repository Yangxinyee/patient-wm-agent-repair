The tensor list settles the structure:
- **Slot update:** `c_out` and `b_in` are plain bias-free linear maps.
- **Coda:** a residual transformer layer over the slots, so `y = c_out(h) + attention/FFN terms`.
- **Normalisation:** `out_norm` and `query_norm` are LayerNorms.

That gives a possible linear route. A dose signal that enters the slots can persist through `c_out` and the per-dimension decays (`flow.a`, `a_bar`), and is read out through `out_norm`. Whether this works depends on three things:
- whether `c_out` is close to the identity, so slot dimensions keep their meaning across steps;
- which dimensions have hour-scale timescales;
- how strongly `out_norm` and `query_norm` pass each dimension.

Next I'll compute these quantities. The hypothesis is confirmed if `c_out` is near-identity and some dimensions have hour-scale decay with non-negligible `out_norm` weight. It is refuted if `c_out` mixes dimensions heavily; I would then use eigen-directions of `c_out·diag(a_bar)` instead.

```python
import numpy as np, torch
C=load('core.c_out.weight').numpy(); I=np.eye(192)
print('c_out: rel dist to I %.3f, diag mean %.3f sd %.3f, offdiag rms %.4f'%(np.linalg.norm(C-I)/np.linalg.norm(I), np.diag(C).mean(), np.diag(C).std(), np.sqrt(((C-np.diag(np.diag(C)))**2).mean())))
fa=load('core.flow.a').numpy(); tau_h=1/np.exp(fa)/3600
print('flow tau hours quantiles', np.round(np.quantile(tau_h,[0,.1,.25,.5,.75,.9,1]),3))
sp=np.log1p(np.exp(load('core.delta_in').numpy())); abar=np.exp(-sp*np.exp(load('core.a_in').numpy()))
print('softplus(delta) q', np.round(np.quantile(sp,[0,.25,.5,.75,1]),3)); print('a_bar q', np.round(np.quantile(abar,[0,.1,.25,.5,.75,.9,1]),3))
for n in ['core.out_norm.weight','query_norm.weight','core.prelude_norm.weight']:
    w=load(n).numpy(); print(n,'abs q', np.round(np.quantile(np.abs(w),[0,.1,.5,.9,1]),3))
ev=np.linalg.eigvals(C*abar[None,:]); print('eig |C diag(abar)| top', np.round(np.sort(np.abs(ev))[::-1][:6],3), 'median', np.round(np.median(np.abs(ev)),3))
```