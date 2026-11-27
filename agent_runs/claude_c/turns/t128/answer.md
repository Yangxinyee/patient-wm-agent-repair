Pass 10 (z_in ×4 along the glucose- and MAP-surprise directions) left factual NLL unchanged: 1.63461 vs 1.63462. Before I decide whether the z-token route is dead, I need to know if the edit moved the predicted means at all. If the predictions barely moved, the z-token has no power over the state and the remap idea is ruled out. Otherwise the predictions moved without hurting NLL.

```python
import pandas as pd, numpy as np
fo=pd.read_csv('/work/fwd_10/factual_original.csv'); fe=pd.read_csv('/work/fwd_10/factual_edited.csv')
print(list(fo.columns)[:12], len(fo))
pc=[c for c in fo.columns if 'mean' in c.lower() or 'pred' in c.lower()][0]
d=(fe[pc]-fo[pc]).abs()
ev=[c for c in fo.columns if 'event' in c.lower() or 'code' in c.lower()][0]
print(pc, ev)
print(d.groupby(fo[ev]).agg(['mean','max']).round(4))
```