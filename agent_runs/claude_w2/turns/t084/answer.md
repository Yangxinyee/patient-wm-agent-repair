The model's one-step behaviour explains its weak dose response.

| Term | Model | Records |
|---|---|---|
| Insulin, lag 1 | −0.31 | −1.99 |
| Insulin, lag 2 | −0.10 | −1.47 |
| Insulin, lag 3 | −0.49 | −0.89 |
| Insulin, lag 4 | ≈0 | +0.17 |
| Insulin, lags 5–8 (positive tail) | +0.3 to +0.5 | +0.3 to +0.8 |
| Glucose persistence (sum of lag coefficients) | 0.87, constant 18 | 0.98 |

- The linear form describes the model's predicted means well (residual sd 3.2).
- The model already has most of the late positive tail. What it lacks is the strong short-lag lowering: about −1.7, −1.4 and −0.4 at lags 1–3.
- The model makes up for the missing insulin effect at high glucose by mean-reverting more strongly. That is why it still fits the records.
- The policy-band terms do not show the absorption I had assumed. The model's band coefficients are more positive than the records', not more negative.

**What this implies for the edit.**
- Adding only the missing dose terms would double count at high glucose, which hurts the factual fit, and under-propagate shifts in the rollout.
- A consistent fix changes the dose terms at lags 1–3 together with glucose persistence.
- The current dose (lag 1) is decodable from the state. That fits the model's state holding dose information that its head barely uses.

**Next step.** Check whether these linearisations predict the rollouts. I will iterate the model's fitted one-step function under insulin x2 on the sel1 stays and compare it with the tool's actual dose_edit result for the original model (−2.14). I will also iterate the variants: model persistence with the records' dose terms, and the records' full function. If the model's linearisation reproduces −2.14 and the lag profile, I can design and calibrate the correction on this linear surrogate before spending forward passes. First I locate my tuning-grid and dose_edit variables.

```python
for nm_ in ['gt','gt2','gtt','gtt2','T2','de','r','dn','A2','fa','fa4']:
    v_ = globals()[nm_]
    print(nm_, getattr(v_, 'shape', None), list(v_.columns)[:24] if hasattr(v_, 'columns') else '')
print(de.head(3).to_string())
print(VOCAB['value_bin_note']); print(VOCAB['numeric_note']); print(VOCAB['numeric'])
print({k_: np.round(v_, 1).tolist() for k_, v_ in VOCAB['value_bin_centres'].items()})
```