Facts about the core's linear pieces:
- 14 h-dimensions have time-flow timescales ≤ 20 min and 34 have ≤ 60 min, so fast-decaying target directions exist.
- `c_out` and `b_in` are dense, with singular values ranging from about 5 down to 0.003.
- softplus(delta_in) ≈ 1, and a_bar ranges 0.28–0.45.

To keep the channel clean without distorting E's LayerNorm, the injected E-component must stay ≲ 5 units, so it has to sit along a direction where E naturally varies very little. Next I'll compute E at every bundle of the 100 stays and examine the eigen-spectrum of its covariance. Eigenvalues with sd ≲ 0.02 would make a clean E-channel feasible; a flat spectrum would rule it out.

```python
EALL, OALL, GALL, KALL, DRG = [], [], [], [], []
for s in sorted(recs.stay_id.unique()):
    B0, rows0, EU0 = stay_data(int(s))
    for k in range(1, len(B0)):
        h0_, E_, y_, o_, att_ = step_parts(k)
        EALL.append(E_.double().mean(0).numpy()); OALL.append(o_.double().numpy()); GALL.append(int(s)); KALL.append(k)
        DRG.append(any(a['code'] in (ci,cn) and a['kind']==0 for a in B0[k][1]))
EALL, OALL, GALL, KALL, DRG = map(np.array, (EALL, OALL, GALL, KALL, DRG))
for nm, X_ in [('E', EALL), ('out', OALL)]:
    ev_ = np.linalg.eigvalsh(np.cov(X_.T)); print(nm, 'sd along eigdirs: smallest', np.sqrt(np.clip(ev_[:6],0,None)).round(4), 'median', np.sqrt(np.median(ev_)).round(4), 'largest', np.sqrt(ev_[-3:]).round(3))
gpn = load('core.prelude_norm.weight').double().numpy(); print('prelude_norm gain quantiles', np.quantile(np.abs(gpn), [0,.1,.5,.9,1]).round(3))
```