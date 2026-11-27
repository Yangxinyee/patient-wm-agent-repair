The original model's factual baseline on these 100 tuning stays is 1.528 nats per value over 5,220 readings:

| reading | NLL | mean abs error |
|---|---|---|
| glucose_finger | 1.575 | 11.7 |
| glucose_lab | 1.287 | |
| map_arterial | 1.359 | |
| map_cuff | 1.754 | |

My edit must not raise the per-value NLL by more than 0.005 on unseen stays, and this is the yardstick I will check against.

Next I will open the per-reading file and join it to my hourly panel. Then I will regress the model's predicted glucose_finger mean one hour after an insulin dose on the dose's jitter (dose minus policy mean), with the current reading as a control, and do the same for the actual readings. If the model's coefficient is near 0 while the data's is about −2, the model never learned the one-step effect. I will also measure how strongly the model's prediction follows the last reading, which matters for how a tilt would propagate in rollouts.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.shape); print(fa.columns.tolist()); print(fa.head(12).to_string())
fa['h'] = fa['hour'].round().astype(int) if 'hour' in fa else None
gfp = fa[fa.event=='glucose_finger'][['stay_id','h','pred_mean' if 'pred_mean' in fa else fa.columns[3]]].copy()
print(gfp.head())
```