The model's implied rollout kernel is about 10% of the truth. For insulin it is about −0.2 per u/h at lags 1–2 and −0.1 to −0.17 out to lag 7, against the true −2.0, −2.36, −2.1, −1.6, … For norepinephrine it is a flat 0.1–0.17 against the true 3.2, 2.5, 1.7, 0.9, 0.3. So the model barely uses the dose, and the shape is too flat as well as too weak.

Since factual records tie dose closely to glucose and MAP, any added effect must act on dose deviations from what the policy would have given, not raw doses. Otherwise factual predictions get biased.

The edit that changes predictions without touching the dynamics is head-only, through z_prior and z_head, reading the dose memory from ctx. That only works if the state still holds recent doses. So next I'll test decodability: ridge-regress, with cross-validation by stay, the insulin and NE rates at lags 1–4 from the 992-number states of the original model.

High R² for lags 1–3 would support a head-side readout. Low R² would mean the dose information fades from the state, so I would have to edit the dynamics instead. I'll also check what `pibar` and the state-index arrays contain.

```python
print('S3', S3[:3], 'hr_', np.round(hr_[:6],3), 'hi_', hi_[:6], 'hn_', hn_[:6], 'nat', nat[:6])
print('pibar test', [pibar(np.array([140.,155,190,230,260]))] if True else '')
```