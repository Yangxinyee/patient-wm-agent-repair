Removing the action token entirely costs only +0.0036 nats of factual NLL (1.63461 → 1.63820). So the model barely relies on drug records for its factual predictions. Next I'll measure how far the predictions actually move without the token. I'll compute the size of the prediction difference and how it scales with the preceding hour's rate. That tells me how much effect the token pathway can carry at its current strength.

```python
import pandas as pd, numpy as np
fo = pd.read_csv('/work/fwd_15/factual_original.csv'); fe = pd.read_csv('/work/fwd_15/factual_edited.csv')
print(fo.columns.tolist()[:12], len(fo))
m = fo.merge(fe, on=[c for c in fo.columns if c in ('stay_id','time','hour','code','event','value')], suffixes=('_o','_e'))
pc = [c for c in fo.columns if 'mean' in c or 'pred' in c]
print(pc)
```