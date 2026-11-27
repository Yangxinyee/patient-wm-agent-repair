The fwd_8 rollout was run with the current edit (hash 45351ba6ba, the same as fwd_9), so its drift and wrong-signed dose response are real properties of v4.

The start-of-stay leak now has a concrete source. Every stay begins with a `hospital_admission` event at hour −2, followed by pre-ICU readings at hours −2 and −1. So the first bundle the model reads is an admission bundle, processed with the tiny initial slot tokens. That is where the `b_in` channel picks up a large spurious signal (+51 mg/dL at the first readings).

The puzzle remains. In teacher-forced mode, v4 changes predictions by only about 0.04 mg/dL once a stay is underway. Yet in the rollout the recorded-dose path is already +4 mg/dL higher at hour 13, the first hidden hour. The prediction at hour 13 comes from the state after the hour-12 drug bundle, which teacher forcing also computes. So the two should agree unless the rollout computes something differently.

To check, I will take the 30 stays that appear in both fwd_8 and fwd_9 and compare, for both models, the hour-13 finger prediction from the rollout with the teacher-forced one.
- If they agree for the original but not for the edited model, the rollout runs the edited model through a path teacher forcing does not exercise. Then the rollout, not the factual run, is the binding test.
- If they agree for both, the hour-13 difference is confined to stays outside the overlap, and I will look at those stays instead.

```python
import numpy as np, pandas as pd
ov = sorted(set(st8) & set(f9.stay_id.unique())); print('overlap stays', len(ov))
r8 = m[(m.edit=='insulin_x2') & (m.stay_id.isin(ov))][['stay_id','hour','factual_o','factual_e','reached']]
t9 = f9[(f9.event=='glucose_finger') & (f9.stay_id.isin(ov))][['stay_id','hour','predicted_mean_o','predicted_mean_e','observed_o']].copy()
t9['hour'] = np.round(t9.hour).astype(int)
cmp_ = r8.merge(t9, on=['stay_id','hour'], how='inner')
for h in [13, 14, 16, 20]:
    cc_ = cmp_[cmp_.hour==h]
    if len(cc_): print('hour', h, 'n', len(cc_), ' |roll_o - tf_o| mean', np.abs(cc_.factual_o-cc_.predicted_mean_o).mean().round(2),
                       ' |roll_e - tf_e| mean', np.abs(cc_.factual_e-cc_.predicted_mean_e).mean().round(2),
                       ' roll drift (e-o) mean', (cc_.factual_e-cc_.factual_o).mean().round(2), ' tf diff (e-o) mean', (cc_.predicted_mean_e-cc_.predicted_mean_o).mean().round(2))
print(cmp_[cmp_.hour==13].head(8).round(1).to_string())
```