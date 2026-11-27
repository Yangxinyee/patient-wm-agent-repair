I need to re-orient before acting. My notebook ends with "Pass 23 … Cleared ops." But the op list now holds 48 rank-1 edits on `core.prelude.0.attn.out.weight` that the notebook never mentions. My last run printed 100 training stay ids. Before using any forward passes, I'll find out what these 48 ops are and what I was preparing. I'll look at their sides, alphas and directions, the recent kernel variables, and `/work/vectors`. If they are 48 basis-vector edits on columns 0–47, they are a head-0 column re-weighting. If the directions are dense, they come from some fitted subspace.

```python
import json, os, numpy as np
o = ops()
print(type(o), len(o)); print(str(o[0])[:300]); print(str(o[-1])[:300])
print(sorted(os.listdir('/work/vectors'))[:40])
try:
    oj = json.load(open('/work/ops.json')); print('ops.json', type(oj), len(oj), str(oj[0])[:300] if len(oj) else '')
except Exception as e: print('ops.json err', e)
g = [k for k,v in globals().items() if not k.startswith('_') and not callable(v) and not isinstance(v, type(os))]
print(len(g)); print(', '.join(g)[-2500:])
```