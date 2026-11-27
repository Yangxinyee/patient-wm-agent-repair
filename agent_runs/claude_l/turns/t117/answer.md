Factual pass 4 gives the original model's baseline on batch1: 1.5578 nats per reading over 4,989 readings. Finger-glucose mean absolute error is 10.5 mg/dL and arterial-MAP error 4.0 mmHg.

Now the test. For each finger-glucose reading I'll regress the model's error on the insulin dose of the previous hour, split into the policy-predicted part and the residual ε. I'll do the same for arterial MAP with norepinephrine. I'll check the CSV columns first and adapt to them.

```python
fc = pd.read_csv('/work/fwd_4/factual_original.csv')
print(fc.shape, list(fc.columns)); print(fc.head(3).to_string()[:600])
```