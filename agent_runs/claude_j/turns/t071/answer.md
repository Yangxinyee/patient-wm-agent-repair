I've been restarted. The last result looks like a check of my CPU replica of the model against the GPU `states` output. The median difference was tiny (4e-7), but the maximum was 0.1 across 1,973 bundles. The table seems to break bundles down by which drug start/end atoms they hold. I don't remember the details, so before going on I'll look at what is in the kernel and re-read the architecture file. If the replica is right except for a few action-atom bundles, the next step is to fix those cases and then design the edit.

```python
import types, os, json
g = {k:v for k,v in globals().items() if not k.startswith('_')}
for k,v in g.items():
    if isinstance(v, types.ModuleType) or k in ('In','Out','exit','quit','get_ipython'): continue
    t = type(v).__name__
    s = ''
    if hasattr(v,'shape'): s = str(getattr(v,'shape'))
    elif isinstance(v,(list,dict,tuple,set,str)): s = f'len {len(v)}'
    print(k, t, s)
print(os.listdir('/work/fwd_3'))
print(open('/model/ARCHITECTURE.md').read())
```