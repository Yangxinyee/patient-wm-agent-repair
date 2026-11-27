Fitting the decoder with zero targets off the gate (ω = 0.1) fixed the extrapolation. Off-gate decoder values stay bounded (99.9th percentile 0.56), and the gated cross-validated R2 is 0.22 with slope k 0.19.

Two other problems remain:
- The cost on gated rows at C = 6 is large: +0.29 nats per row. This matches theory. An unbiased decoder adds prediction variance of C²·Var(D)/R2, about 0.31·C². Keeping the cost near 0.0017 overall limits C to about 1.8 mmHg per unit, roughly half the true lag-1 effect.
- Glucose rows lose about +0.011 regardless of ω. That cost comes from ablating units 54 and 9 with only a global-mean bias compensation, so the old "+0.0003" ablation figure does not hold for this compensation.

Before any forward pass, I want a cheaper host for the channel. Plan:
- Scan single-unit ablation costs to find the cheapest units.
- Check whether `embed_transition.value_bin`, a per-(code, last-bin) logit table, can carry code-specific compensation. That would make repurposing nearly free.
- Confirm that HI holds the per-row transition logits I need for an exact offline evaluator.

```python
print([n for n in NAMES if 'transition' in n], [SHAPES[n] for n in NAMES if 'transition' in n])
print('HI' in globals(), (list(HI.keys()) if 'HI' in globals() and isinstance(HI, dict) else type(globals().get('HI')).__name__))
Hall = gel(WCd@W1c.T + b1c); mall = Hall.mean((0,1))
cost = np.full(192, np.nan)
for i in range(192):
    if i in (31,143): continue
    W2a = W2c.copy(); b2a = b2c + W2c[:,i]*mall[i]; W2a[:,i] = 0
    cost[i] = float((eval_E(W1c,b1c,W2a,b2a)[0]-rE[0]).mean())
o = np.argsort(cost); print('cheapest', [(int(i), round(cost[i],5)) for i in o[:12]])
print('54,9', round(cost[54],5), round(cost[9],5), 'median', round(float(np.nanmedian(cost)),5))
```