The architecture is clear in outline. Doses enter only through the action encoder (u_k), via z_dose/z_rate, the dose and rate bins and their has-flags. A dose edit can therefore change glucose predictions only through the state (z posterior, prelude, recurrent block, out). It cannot act through the glucose "last value" register. Next I will look at the vocabulary (normalisation, bin edges), the tensor list and the raw records. That tells me how doses are encoded and how dosing relates to readings. Nothing to confirm or refute yet; I am still orienting.

```python
import numpy as np, pandas as pd
print(json.dumps({k:(v if not isinstance(v,(list,dict)) or len(str(v))<3000 else str(v)[:3000]) for k,v in VOCAB.items()}, indent=1)[:8000])
print(len(NAMES))
for n in NAMES: print(n, SHAPES[n])
```