The last run (125) failed with a `KeyError` on the id 1010000000, probably a stay-id lookup. My notebook stops at run 99, so runs 100–125 are not recorded. I don't know what code ran in 125 or which kernel variables are current. Before doing anything else I will take stock of the kernel: its variables, functions and the files in /work. I will also read the start of the notebook. Then I can rebuild the plan without repeating work or trusting stale variables. A good result is a clear inventory: the exact-replica functions (`run_stay`, `core_step`, `zpost_of`, ...), the perturbed-sample data, and any decoders or coefficient estimates from runs 100–125.

```python
import types, inspect, os, json
g = dict(globals())
rows = []
for k, v in g.items():
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    t = type(v).__name__
    info = ''
    try:
        if callable(v) and isinstance(v, types.FunctionType):
            c = v.__code__
            info = f"args={c.co_varnames[:c.co_argcount]} lines={c.co_firstlineno}"
        elif hasattr(v, 'shape'): info = str(tuple(v.shape))
        elif isinstance(v, (list, tuple, dict, set)): info = f'len={len(v)}'
        elif isinstance(v, (int, float, str, bool)): info = repr(v)[:80]
    except Exception as e:
        info = '?'
    rows.append((t, k, info))
for r in sorted(rows):
    if r[0] in ('type','builtin_function_or_method'): continue
    print(r)
print('----- files')
for root, dirs, files in os.walk('/work'):
    for f in files:
        p = os.path.join(root, f); print(p, os.path.getsize(p))
print('----- notes head')
s = open('/work/notes.md').read()
print(len(s))
print(s[:max(0, len(s)-8000)][:6000])
```