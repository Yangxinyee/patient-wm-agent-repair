No channel of h is dead. The smallest per-channel sd is 2.7 against a median of 7.9, and the four slots have almost identical statistics. So "take over an unused channel" is not an option. The state is large-scale (sd ~8 per channel), and its decay and input-gain parameters (`core.flow.a`, `core.a_in`, `core.delta_in`) are per channel with 192 entries.

A cleaner alternative is to write a dose-innovation signal along a direction where factual states barely vary (a smallest principal component). The head could then read it with high signal-to-noise. The signal would only stay put in that direction if its channels share similar decay rates. I'll check the eigen-spectrum of slot-0 h on factual states and the spread of the per-channel time decay (at 1 h) and per-step retention. A very low-variance subspace, plus groups of channels with near-identical decay, would support this design. A flat spectrum would push me toward a head-only decoder.

```python
fa = load('core.flow.a'); ai = load('core.a_in'); di = load('core.delta_in')
sp_ = np.log1p(np.exp(di)); abar = np.exp(-sp_*np.exp(ai)); dec1h = np.exp(-3600.0*np.exp(fa)) if np.exp(fa).max()<1e-2 else np.exp(-1.0*np.exp(fa))
print('flow.a range %.2f..%.2f' % (fa.min(), fa.max()), 'exp(a) min/max', np.exp(fa).min(), np.exp(fa).max())
print('abar per step: q05 %.3f med %.3f q95 %.3f min %.3f max %.3f' % (*np.quantile(abar,[.05,.5,.95]), abar.min(), abar.max()))
print('decay per 1h (if dt in hours): q05 %.3f med %.3f q95 %.3f' % tuple(np.quantile(np.exp(-1.0*np.exp(fa)),[.05,.5,.95])))
print('softplus(delta_in) range %.3f..%.3f' % (sp_.min(), sp_.max()))
H0 = X[:, 192:384]; Hc = H0 - H0.mean(0); ev = np.linalg.svd(Hc, compute_uv=False)**2/len(Hc)
print('h slot0 PCA sd: top5', np.sqrt(ev[:5]).round(2), ' last8', np.sqrt(ev[-8:]).round(3), ' n(sd<0.5)=', int((np.sqrt(ev)<0.5).sum()))
O = X[:, :192]; Oc = O - O.mean(0); evo = np.linalg.svd(Oc, compute_uv=False)**2/len(Oc)
print('out PCA sd: top3', np.sqrt(evo[:3]).round(2), 'last5', np.sqrt(evo[-5:]).round(3))
```