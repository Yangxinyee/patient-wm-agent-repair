Finger and lab glucose agree with no bias: mean difference -0.03, SD of the difference 9.0, so each carries roughly 6 mg/dL of independent noise if the errors are similar. Cuff MAP reads about 3 mmHg above arterial, SD of the difference 6.7. Finger readings are taken 97% of the hours on insulin vs 31% off it.

This opens a clean negative-control test for confounding. A dose given at t+5 min cannot causally affect a reading taken at hour t. So if insulin_t predicts lab_t once finger_t (and the rest of the recorded history) is held fixed, clinicians are dosing on information the record does not contain. That would mean "true glucose" confounding, and the model's dose response would be biased by it.

I'll fit the insulin policy with a flexible regressor on recorded history, look at how much variation remains, and then test whether the policy residual predicts the same-hour lab glucose.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
d['g_prev'] = d.groupby('stay_id').glucose_finger.shift(1)
d['g_prev2'] = d.groupby('stay_id').glucose_finger.shift(2)
d['gl_prev_last'] = d.groupby('stay_id').gl_last.shift(1)
# rows: fresh finger at t
X_cols = ['glucose_finger','g_prev','g_prev2','ins_prev','gl_prev_last','ih','ne_prev','mc_last','ma_last']
m = d[d.glucose_finger.notna()].copy()
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1)
idx_tr = m.stay_id % 5 != 0
gb.fit(m.loc[idx_tr, X_cols], m.loc[idx_tr,'insulin'])
m['ins_hat'] = gb.predict(m[X_cols])
res = (m.insulin - m.ins_hat)[~idx_tr]
print("R2 holdout:", 1 - res.var()/m.insulin[~idx_tr].var(), "resid sd", res.std())
m['ires'] = m.insulin - m.ins_hat
# negative control: does residual predict same-hour lab glucose given finger?
mm = m[m.glucose_lab.notna() & ~idx_tr]
import statsmodels.api as sm if False else None
```