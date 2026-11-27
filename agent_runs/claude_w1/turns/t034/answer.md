The original model's baseline on these 100 stays is 1.519 nats per value over 5,177 values. The NLL is a 16-bin classification loss. By reading type:

| Reading | NLL (nats) | Mean abs error |
|---|---|---|
| Finger glucose | 1.531 | 10.8 mg/dL |
| Lab glucose | 1.261 | 8.6 mg/dL |
| Arterial MAP | 1.349 | 4.0 mmHg |
| Cuff MAP | 1.746 | 6.2 mmHg |

Next I'll look at the per-reading file, then regress the residuals (actual minus predicted mean) on insulin dose shocks at lags 0–5:
- **Shock coefficients near −2** would mean the model misses the causal dose effect.
- **Near 0** would mean it captures it factually, and only the rollout response is wrong.

I'll also test whether residuals depend on the expected dose (the base part). If they don't, the model has fully absorbed the average effect into its glucose terms. Then the edit must target only deviations from the expected dose.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.shape, fa.columns.tolist()); print(fa.head(12).to_string())
fa['h'] = np.floor(fa.hour).astype(int) if 'hour' in fa else None
```