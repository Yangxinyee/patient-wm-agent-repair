I have the pieces to study the rollout locally: `make_bundles`, `run_tf`, `mystep`, `scale_atoms` and the head functions, plus the tuning DataFrame `tun`. Before building a local hour-13 check, I'll use data already in hand.

fwd_4 has the tool's per-stay-hour shifts. For the edited model, the x2 shift at hour 13 should track the stay's hour-12 insulin rate at about −1.3 per u/h if the edit responds to the dose in the tool's rollout. The difference between the edited and original factual rollouts tells me whether the edit fires at all.

- If the hour-13 x2 shifts are flat (about 0 whatever the rate), the tool's dose-edited input is not reaching my decoder the way my local simulation of x2 assumes.
- If they do scale with rate but later hours wash it out, the problem is the dynamics.

```python
import pandas as pd, numpy as np
d4e = pd.read_csv('/work/fwd_4/dose_edit_edited.csv'); d4o = pd.read_csv('/work/fwd_4/dose_edit_original.csv')
print(tun.columns.tolist()); print(tun.head(4).to_string())
print(d4e.groupby('edit').apply(lambda g: pd.Series({'h13': g[g.hour==13]['shift'].mean(), 'all': g[g.reached]['shift'].mean(), 'n13': (g.hour==13).sum()})))
# factual edited - original by hour
m = d4e.merge(d4o, on=['stay_id','hour','edit'], suffixes=('_e','_o'))
mm = m[m.edit=='insulin_x2']
print('factual diff (edited-orig) by hour:', mm.groupby('hour').apply(lambda g: (g.factual_e-g.factual_o).mean()).round(2).to_dict())
print('x2 shift edited by hour:', mm.groupby('hour').shift_e.mean().round(3).to_dict())
print('x2 shift orig by hour:', mm.groupby('hour').shift_o.mean().round(3).to_dict())
```