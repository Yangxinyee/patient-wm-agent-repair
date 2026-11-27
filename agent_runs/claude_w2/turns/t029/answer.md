The baseline factual fit on these 100 tuning stays is a value NLL of 1.5279 per reading (5,238 readings). Mean absolute error is 10.4 mg/dL on glucose_finger, 8.4 on glucose_lab, 3.9 mmHg on map_arterial and 6.2 on map_cuff. So my edit must keep the value NLL below about 1.5329 on comparable stays.

Next I'll look at the factual CSV. For each glucose_finger reading that follows an insulin record in the previous hour, I'll regress the model's predicted mean on that insulin rate, within glucose band and controlling for the previous reading. I'll do the same for MAP on the NE rate. A model slope well below the data's −2 mg/dL per U/h (or +3.6 mmHg per unit NE) means it under-uses the dose even in teacher-forced prediction. Then amplifying the dose pathway should improve factual NLL rather than hurt it. A slope near −2 would point the problem at rollout dynamics instead.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.shape, fa.columns.tolist()); print(fa.head(8).to_string())
```