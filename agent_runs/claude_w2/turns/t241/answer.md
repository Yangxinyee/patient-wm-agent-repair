The local engine can record the head inputs at every prediction (base, last-value features, last bin, prior mean of z) via `store=`. It also computes teacher-forced NLL with `want_nll=True, t_hide=1e9`. So I can test a weight edit of the value head entirely locally, by passing an edited weights dict.

Before fitting decoders I need two numbers I never recorded on train data: what the original model's factual errors depend on.

- **Insulin:** whether the residual depends on the deviation of the dose from the policy only (meaning the model has folded the average insulin effect into its glucose dynamics), or on the absolute dose.
- **Norepinephrine:** whether the residual depends on the innovation or on the level.

This decides the compensation term of the edit. I'll check the residual table's columns and run compact OLS fits for finger glucose and arterial MAP.

```python
print(M2.shape, [c for c in M2.columns][:60])
```