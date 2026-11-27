The valid results change the picture:
- **Dose memory in the head input is short.** Lag-1 dose moves the head input by |a₁| = 0.043 per u/h, but lags 2–3 by only 0.007 and 0.002. The model's state forgets the dose within about an hour, so a head-only decoder can serve at most lags 1–2.
- **Centring is feasible.** Factual doses are well predicted from the head input. The lag-1–3 residuals have sd 0.49–0.58, about the innovation sd, and correlate 0.80–0.83 with the true innovations. So "expected dose given history" is linearly available.

What I need next is the missing direct effect at each lag in the model's own frame: regress (finger reading − model's predicted mean) on lagged innovations. If it is concentrated at lags 1–2, a head-level edit can work. For that I need the predicted mean computed from these head-input vectors. I'll locate the value-head and transition tensors and the tuning DataFrame, so I can verify my head computation against the tool's factual means in `fwd_2`.

```python
print([ (n, SHAPES[n]) for n in NAMES if 'value_bin' in n or 'transition' in n][:12])
tdfs = [(k, v.shape) for k, v in globals().items() if isinstance(v, pd.DataFrame) and 'stay_id' in v.columns and v.shape[0] < 100000]
print('small dfs with stay_id:', tdfs[:8])
print('fwd_2 files:', os.listdir('/work/fwd_2'))
f2 = pd.read_csv('/work/fwd_2/' + [f for f in os.listdir('/work/fwd_2') if f.startswith('factual')][0])
print(f2.columns.tolist(), f2.shape); print(f2.head(3).to_string()[:500])
```