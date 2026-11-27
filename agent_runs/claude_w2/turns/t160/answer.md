Recovered the structural kernels I estimated earlier from the records (effect per unit rate-hour at lags 1–12):
- insulin → glucose: −2.01, −2.49, −2.24, −1.80, −1.42, −1.15, −0.94, −0.75, −0.50, −0.42, −0.33, −0.27;
- norepinephrine → MAP: +3.60, +2.73, +1.72, +1.08, +0.42, +0.45, then ~0.

These define the true counterfactual shift for any readout row: Σ K(lag) × (F − 1) × rate of each edited infusion before the reading.

For the head correction I also need the one-step structural coefficients: next reading regressed on its own lags, dose lags and policy terms. I need them for both insulin and NE, plus the model's own one-step dose response for comparison. I'll look at the stored training grid `gg` to rebuild those regressions.

```python
print(gg.columns.tolist())
print(gg.iloc[:3].T.head(60))
```