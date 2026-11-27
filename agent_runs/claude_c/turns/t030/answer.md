The baseline factual fit on these 100 tuning stays is a value NLL of 1.6346 nats per reading over 4,751 readings.

| Reading | MAE | NLL (nats) |
|---|---|---|
| Finger glucose | 10.8 mg/dL | 1.46 |
| Lab glucose | 8.6 mg/dL | 1.29 |
| Arterial MAP | 3.8 mmHg | 1.47 |
| Cuff MAP | 6.3 mmHg | 2.01 |

The NLL is over 16 fine bins, so it is discrete. The 0.005-nat budget is small against a per-reading glucose sd of roughly 13 mg/dL.

Next I will regress the model's teacher-forced predicted mean for the next finger glucose on the within-tier insulin dose, controlling for the current reading, tier and earlier readings. I will run the same regression on the actual readings for comparison. A model slope near −2 means it learned the causal effect; near 0 means it did not. I will do the same for NE on the next arterial MAP.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.columns.tolist()); print(fa.head(10).to_string())
print(fa.event.value_counts() if 'event' in fa.columns else '')
```