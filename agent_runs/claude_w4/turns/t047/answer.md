Clean result: the old model ignores dose surprises in factual mode. Regressing its factual glucose_finger residuals on insulin surprises gives γ = [−2.05, −1.64, −1.33, +1.1, …] for lags 1–4, with standard errors around 0.6–0.7. The one-hour coefficient matches the true −1.83 effect, so the model leaves the effect unexplained. The glucose_lab coefficients point the same way but are noisier. The surprise lags improve R² by about 1.4%, so adding the effect should slightly improve factual NLL and give me slack under the 0.005 limit.

The edit I want: after an insulin bundle, the model's glucose belief should move by γ × (rate − expected rate for the band). The effect must persist over a few hours; the model's own handling of its drawn readings can carry the rest. An analogous edit is needed for norepinephrine on MAP. To build this through attention and norm layers, I need a CPU copy of the model checked against the fwd_3 states.

First step: the tokenisation details. I will print:
- the full code list and the numeric entries for rate and dose bins;
- whether the first few bundles' states are identical across stays (they should be if bundle 0 is always a bare hospital_admission at −2 h);
- the state values for the first bundles, as fixed points to check the copy against.

```python
print(json.dumps(VOCAB['codes']))
for e_ in VOCAB['numeric']:
    print(e_['code'], e_['attribute'], e_['median'], e_['iqr'], e_['fine_bin_edges'])
X_ = S['x']; sid_ = S['stay_id']; hr_ = S['hour']
first = np.r_[0, np.flatnonzero(np.diff(sid_)!=0)+1]
print('n stays in states', len(first), ' first hours', np.unique(hr_[first]))
x0 = X_[first]; print('bundle0 spread across stays (max abs dev):', np.abs(x0 - x0[0]).max())
x1 = X_[first+1]; print('bundle1 hours', np.unique(hr_[first+1])[:5], ' spread', np.abs(x1-x1[0]).max())
print('out0[:8]', np.round(x0[0,:8],4)); print('z0[:8]', np.round(x0[0,960:968],4))
print('h slot0 [:6]', np.round(x0[0,192:198],4), ' norms per slot', [round(float(np.linalg.norm(x0[0,192+192*s:384+192*s])),3) for s in range(4)])
```