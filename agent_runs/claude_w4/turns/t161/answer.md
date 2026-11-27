The architecture description is back in view, and it matches my notebook:
- Dose enters only through u_k.
- u_k feeds the prelude transformer layer.
- The z token is barely used, which is why v1 failed.
- E, the prelude's slot outputs, drives the slot update through b_in(E) and the recurrent blocks.

The variable listing was cut off at 10,000 characters before the newest objects. Those newest objects are the ones that would show what runs 123 to 139 did.

Next I will:
- print only the globals defined after the cut point;
- if the kernel keeps an input history, print the code of the last few runs (shortened), so I can see what the lnM / ageKV / noage comparison in run 139 was.

What would confirm my working assumption: objects such as a CPU copy of the prelude or of the recurrent block, and a candidate head-1 value edit. That would mean runs 123 to 139 were building the planned v2 injection, and I can resume it.

```python
import types
items = [(k, v) for k, v in list(globals().items()) if not k.startswith('_')]
keys = [k for k, _ in items]
start = keys.index('unit') + 1 if 'unit' in keys else len(keys) - 300
desc = []
for k, v in items[start:]:
    if isinstance(v, types.ModuleType) or k in ('In', 'Out', 'exit', 'quit', 'get_ipython', 'desc', 'items', 'keys', 'start'):
        continue
    t = type(v).__name__
    if hasattr(v, 'shape'):
        s = str(tuple(v.shape))
    elif isinstance(v, (list, tuple, dict, set)):
        s = f'len{len(v)}'
    elif isinstance(v, (int, float)):
        s = repr(v)[:10]
    elif isinstance(v, str):
        s = repr(v)[:20]
    else:
        s = ''
    desc.append(f'{k}<{t}{":" + s if s else ""}>')
print(len(desc), 'later globals')
print(' '.join(desc))
hist = globals().get('In', None)
print('In exists:', hist is not None, len(hist) if hist is not None else '')
if hist is not None:
    for j in range(max(1, len(hist) - 4), len(hist) - 1):
        src = hist[j]
        print(f'----- cell {j} ({len(src)} chars) -----')
        print(src[:1800])
```