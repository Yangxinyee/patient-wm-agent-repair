The original model's factual fit on these 100 training stays is a value NLL of 1.4404 per reading over 5,684 readings. Per event:

| event | readings | NLL | mean abs error |
|---|---|---|---|
| glucose_finger | 1,865 | 1.553 | 11.0 mg/dL |
| glucose_lab | 541 | 1.292 | 9.2 mg/dL |
| map_arterial | 2,481 | 1.342 | 4.0 mmHg |
| map_cuff | 797 | 1.583 | 5.6 mmHg |

The tolerance is 0.005 nats averaged over all readings. glucose_finger is about a third of them, so an edit that only touches glucose_finger could cost up to about 0.015 nats on those readings.

Next I will look at the factual CSV's columns, so I can match each reading to the stored state just before it and start a local implementation of the value head.

```python
fac = pd.read_csv('/work/fwd_3/factual_original.csv')
print(fac.shape); print(fac.columns.tolist()); print(fac.head(8).to_string())
```