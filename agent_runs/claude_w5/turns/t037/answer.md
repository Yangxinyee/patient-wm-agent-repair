The decoding test settles where dose information lives. Cross-validated R² from the state just before each glucose reading:

| target | R² from `out` | R² from full state |
|---|---|---|
| last glucose reading | 0.96 | 0.97 |
| raw recent insulin rate | about 0.6 | 0.7 |
| insulin rate residual (random part) | negative | 0.22 |
| norepinephrine rate residual | about 0 | 0.14 |

The raw insulin rate is decodable mainly because it is tied to glucose. The random part, the only part the edit may act on, is barely in the state.

So an output-side readout cannot work: the core almost discards the dose values. The edit must make dose information enter and persist in the recurrent state, with a controlled effect on later glucose and MAP predictions.

To find a path with predictable propagation, I'll examine the core's recurrence weights. The questions are:
- whether `core.c_out` is near the identity;
- the ranges of the decay parameters `a_in`, `delta_in` and `flow.a`, giving each dimension's timescale.

If some dimensions behave like simple leaky integrators, a dose signal written there would decay in a controlled way.

```python
c_out = load('core.c_out.weight').numpy() if hasattr(load('core.c_out.weight'),'numpy') else np.asarray(load('core.c_out.weight'))
def L(n):
    t = load(n); return t.numpy() if hasattr(t,'numpy') else np.asarray(t)
c_out = L('core.c_out.weight'); a_in = L('core.a_in'); d_in = L('core.delta_in'); fa = L('core.flow.a')
print("c_out diag mean", np.diag(c_out).mean(), " offdiag abs mean", np.abs(c_out - np.diag(np.diag(c_out))).mean(), " ||c_out - I||/||I||", np.linalg.norm(c_out-np.eye(192))/np.sqrt(192))
ev = np.linalg.eigvals(c_out); print("c_out |eig| range", np.abs(ev).min(), np.abs(ev).max())
sp = np.log1p(np.exp(d_in)); abar = np.exp(-sp*np.exp(a_in))
print("softplus(delta_in) range", sp.min(), sp.max(), " abar range", abar.min(), abar.max())
tau = 1/np.exp(fa); print("flow tau (s) quantiles", np.quantile(tau,[0,0.1,0.25,0.5,0.75,0.9,1]))
print("tau hours sorted (first 30):", np.round(np.sort(tau)[:30]/3600,3))
hm = X[:, 192:960].reshape(-1,4,192)
print("per-dim mean |h| top10 dims:", np.argsort(-np.abs(hm.mean(axis=(0,1))))[:10], np.round(np.sort(-np.abs(hm.mean(axis=(0,1))))[:10],1))
print("per-dim sd of h (over bundles) quantiles:", np.quantile(hm.std(axis=0).mean(axis=0),[0,0.5,0.9,1]))
```