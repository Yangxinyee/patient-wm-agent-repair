This is the key result. The exact linear correction, built from the factual-residual coefficients, brings the insulin rollouts close to the kernel truth on all arms:

| Insulin arm | Corrected model | Truth | Ratio |
|---|---|---|---|
| x2 | −19.4 | −16.5 | 1.17 |
| x½ | +9.8 | +8.3 | 1.18 |
| x0 | +16.9 | +16.5 | 1.02 |
| x3 | −36.6 | −33.0 | 1.11 |
| x2 from hour 16 | −13.7 | −14.2 | 0.97 |

Lags 1–6 match closely; lags 9–12 overshoot. So the deviation design is right for insulin, possibly with a slightly lower gain or extra positive later-lag terms.

For norepinephrine the same recipe does not work. It gives x2 +2.3 against a truth of +5.6, and the effect is flat after lag 3. The fitted coefficients nearly cancel at steady state, and the model keeps pulling MAP back toward the controlled range, as if titration still acted. Norepinephrine will need a separate, sustained-level design later.

Next I implement the insulin correction as real weights. The correction is r = Σa·d + Σ(c−a)·p(g), for finger and lab glucose only. I'll put it in one gated hidden unit of the value head that reads a linear decoder of r from the head input (`with_code`). The decoder is fitted on teacher-forced head inputs under both recorded and edited insulin doses, so it tracks counterfactual doses. For that I need the train-records dataframe, the stay lists and the rollout result format. A short lookup:

```python
rest = sorted([k for k,d in items if k > 'Wc'])
print(' | '.join(f'{k}:{dict(items)[k]}' for k in rest)[:2500])
print('res_b keys', list(res_b.keys()) if isinstance(res_b,dict) else type(res_b))
print({k: len(v['stays']) for k,v in globals().items() if isinstance(v,dict) and 'stays' in v and 'A' in v})
```