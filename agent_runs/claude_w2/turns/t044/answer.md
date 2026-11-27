The helper functions cannot be listed: they were defined dynamically. The states file (fwd_3) covers 100 training stays, all 5,345 bundles. Each bundle has its stay, hour, insulin/NE flags and the 992-dimensional state.

I still hold the structural kernels estimated from the records:
- **Insulin, glucose change per U/h at lags 1–12:** −2.0, −2.5, −2.2, −1.8, −1.4, … The total is about −14.
- **Norepinephrine, MAP change per mcg/kg/min at lags 1–12:** +3.6, +2.7, +1.7, +1.1, +0.4, … The total is about +10.

Next I want to test whether the z posterior mean at action bundles tracks the dose's departure from the policy-expected dose. To do that I need to line the state rows up with the event table. So this run only checks the column layout of the training event table and how bundle hours are coded. A confirming outcome for later: z at insulin bundles should correlate strongly with (rate − expected rate) and weakly with the expected rate.

```python
print(tr.columns.tolist()); print(tr.head(4).to_string())
print(open('/data/README.md').read()[:1800])
hi_ = S['hour'][S['has_insulin']==1][:8]; print(hi_, S['stay_id'][:3], S['n_atoms'][:10])
print(tr[(tr.stay_id==S['stay_id'][0])].head(12).to_string())
```