Over all training stays, my per-dose effect profiles imply these target readouts, assuming linear superposition of recorded doses:

| readout | target |
|---|---|
| insulin x2 | −14.7 mg/dL |
| insulin x1/2 | +7.3 mg/dL |
| norepinephrine x2 | +4.5 mmHg |
| norepinephrine x1/2 | −2.3 mmHg |

Sampling noise across 400-stay sets is small (sd 0.4 and 0.2), so the unseen evaluation stays should give similar truths.

Next I need to know whether the model's state already carries dose information, since that decides between an output-side and a core-side edit. I'll select 100 training stays rich in both drugs (insulin and norepinephrine from hour 12, arterial readings). In the following reply I'll request their states for all bundles. If a linear readout of the state predicts recent doses well (high R²), an output-side edit is feasible. If not, the core must be edited.

```python
tr_ins12 = P[(P.h>=12)&P.insulin.notna()].groupby('stay_id').size()
tr_ne12 = P[(P.h>=12)&P.norepinephrine.notna()].groupby('stay_id').size()
tr_art = P[(P.h>=13)&P.map_arterial.notna()].groupby('stay_id').size()
both_tr = sorted(set(tr_ins12.index)&set(tr_ne12.index)&set(tr_art.index))
rng = np.random.default_rng(1)
selB = [int(x) for x in rng.choice(both_tr, 100, replace=False)]
json.dump(selB, open('/work/sel_train_B.json','w'))
print(len(both_tr)); print(json.dumps(selB))
```