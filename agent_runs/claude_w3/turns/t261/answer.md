Restarted session. The last run only printed the value scaling for the two pressure codes. I don't know how far the edits got: the list shows many tensors edited (consistent with a "dedicated dimension" design), but the notebook does not say whether they were finished or checked. Next I'll take stock of the kernel variables and of how each edited tensor differs from the original.

This should tell me which dimension(s) are reserved and whether the injection and readout paths exist yet. If only the "zero writes" part is present, the injection and readout still need building.

```python
import numpy as np, json, os, types
g = {k: type(v).__name__ for k, v in globals().items() if not k.startswith('_')}
funcs = sorted([k for k, t in g.items() if t == 'function'])
others = sorted([(k, t) for k, t in g.items() if t not in ('function', 'module', 'type', 'builtin_function_or_method')])
print("FUNCS:", funcs)
print("VARS:", [k for k, t in others][:150])
print(os.listdir('/work/edited')[:20] if os.path.isdir('/work/edited') else None)
for n in edited():
    a = load(n); b = load_original(n)
    d = np.abs(a - b)
    if a.ndim == 2:
        rows = np.where(d.max(1) > 0)[0]; cols = np.where(d.max(0) > 0)[0]
        print(n, a.shape, "rows", rows[:8].tolist(), len(rows), "cols", cols[:8].tolist(), len(cols))
    else:
        idx = np.where(d > 0)[0]
        print(n, a.shape, "idx", idx[:8].tolist(), len(idx), "vals", a[idx[:4]].round(3).tolist())
```