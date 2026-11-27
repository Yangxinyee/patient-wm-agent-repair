Run 78 shows the kernel still holds my partial re-implementation helpers (`lin`, `mlp`, `ln`, `atom_vec`, `set_enc`, `act`), the panel data, and a factual forward pass of the original model (fwd_4). The factual results are what I need to size the problem. I want the per-reading residual spread, which tells how much a centring error in an edit would cost in NLL, and the baseline NLL for the 0.005 tolerance. I will print only summary statistics.

```python
import json, pandas as pd, numpy as np
s4=json.load(open('/work/fwd_4/summary.json'))
print({k:(v if not isinstance(v,(list,dict)) else type(v).__name__+str(len(v))) for k,v in s4.items()})
f4=pd.read_csv('/work/fwd_4/factual_original.csv')
print(f4.columns.tolist(), len(f4))
```