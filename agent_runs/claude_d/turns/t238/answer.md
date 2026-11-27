Run 211 showed that edit E3b passes the offline checks. E3b is the insulin readout channel scaled to C = -2.79. The overall factual NLL rises by only +0.0004, or +0.0016 on held-out stays. The bf16 bias is negligible. E3b is therefore a viable insulin edit, and the next forward test should bring insulin x2 to about the -14 target on sel1.

The norepinephrine readouts are still unaddressed. The model gives about 0 against targets of +4.85 and -2.43. Next I will check that the saved edit equals E3b, record this in the notebook, and list the helpers and arrays already in the kernel so I can reuse them for the norepinephrine work. If the saved edit matches E3b with a maximum difference near 0, I can build on it.

```python
import numpy as np, types, pandas as pd
ed = edited(); E = np.load('/work/E3b.npz')
print(ed, E.files)
for k in E.files:
    if k in ed: print(k, float(np.abs(np.asarray(load(k))-E[k]).max()))
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 8 (from run 212)\n- Run 211: E3b (C=-2.79) offline dNLL all +.0004, glu&pres +.0098, map -.0036, held-out +.0016; bf16 bias -.02.\n- Keep replies/outputs short (filter).\n")
g = dict(globals())
fn = sorted(k for k,v in g.items() if isinstance(v, types.FunctionType) and not k.startswith('_'))
arr = sorted((k, tuple(v.shape)) for k,v in g.items() if isinstance(v, np.ndarray) and v.size>1000)
dfs = sorted((k, v.shape) for k,v in g.items() if isinstance(v, pd.DataFrame))
print('FN', ' '.join(fn))
print('ARR', ' '.join(f"{k}{s}" for k,s in arr))
print('DF', ' '.join(f"{k}{s}" for k,s in dfs))
```