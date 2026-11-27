That was far more data than I meant to use: `SIDS` holds thousands of training stays, not 100. The result is clear all the same. On 48k insulin and 71k norepinephrine bundles, my local edited model responds linearly and symmetrically to single-bundle dose changes:

| Drug | Slope dx/drate | Slope of injection on vbar (f < 1 and f > 1) |
|---|---|---|
| insulin | 0.90–1.01 | −3.7 to −4.3 |
| norepinephrine | 0.84–0.91 | +6.4 to +7.1 |

The gate is solidly open at drug bundles (s ≥ 0.86). So the asymmetry in fwd_6 (x1/2 insulin with the wrong sign) does not come from the single-step mechanism as I model it locally.

Two explanations remain:
- the tool's single-step response still differs from mine, for example through float32 precision with the large kv weights;
- the asymmetry builds up over the rollout, through nonlinear dynamics or drift.

Hour 13 separates them, because its prediction comes almost directly from the first edited bundle.
- If hour 13 is symmetric (x1/2 ≈ −½ × x2) and the asymmetry appears later, the cause is rollout dynamics.
- If hour 13 is already asymmetric, the tool's step differs from mine.

I'll also check the edited model's factual drift relative to the original.

```python
import pandas as pd, numpy as np
d6e = pd.read_csv('/work/fwd_6/dose_edit_edited.csv'); d6o = pd.read_csv('/work/fwd_6/dose_edit_original.csv')
m6 = d6e.merge(d6o, on=['stay_id','hour','edit'], suffixes=('_e','_o'))
m6r = m6[m6.reached_e]
tab = m6r.groupby(['edit','hour']).shift_e.mean().unstack(0).round(2)
print(tab.to_string())
fx = m6r[m6r.edit=='insulin_x2']
print('factual diff edited-orig by hour (glucose):', fx.groupby('hour').apply(lambda g: (g.factual_e-g.factual_o).mean()).round(2).to_dict())
fn_ = m6r[m6r.edit=='ne_x2']
print('factual diff edited-orig by hour (MAP):', fn_.groupby('hour').apply(lambda g: (g.factual_e-g.factual_o).mean()).round(2).to_dict())
print('mean shift_se edited by edit:', m6r.groupby('edit').shift_se_e.mean().round(3).to_dict())
```