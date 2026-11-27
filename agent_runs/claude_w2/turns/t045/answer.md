The event table layout is clear:
- one row per event;
- readings at integer hours;
- infusion records at h + 0.0833 with rate, dose and end time.

The state rows from fwd_3 carry the same fractional hours, so I can join each action bundle to its recorded rate and to the glucose reading the policy used just before it.

Next step: find the hourly grids I built earlier (`g` for training, `gt` for tuning, `gfm`/`mam` for factual output merged with doses). I will reuse them instead of rebuilding.

The test I want: regress the original model's teacher-forced glucose residuals on two things:
- the dose's departure from the policy-expected dose;
- the policy-expected dose itself.

If the model absorbed part of the drug effect into its glucose dynamics, the residual should load on the departure (about −1.5 per U/h at lag 1) and not on the expected dose. If it instead loads on both, a plain amplification of the dose pathway would be right.

```python
for nm_ in ['g','gt','gtt2','gfm','mam','fa','m','de','r']:
    v_ = globals()[nm_]; print(nm_, v_.shape, list(v_.columns)[:40])
print(gfm.head(3).to_string()); print(g.head(3).to_string())
```