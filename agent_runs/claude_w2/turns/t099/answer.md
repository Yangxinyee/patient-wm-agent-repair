I'm resuming after a restart. Run 93 compared my partial local re-implementation of the model's core step against the recorded states from the tool (fwd_3) for several bundle types and variants of the action encoding. The errors look like relative errors: about 0.01 to 0.04 for reading bundles, 0.017 for action bundles and about 0.2 for end-only bundles. The best variant was "e_route", with action error 0.0024 and end-only error 0.0039. So the variant that routes the event embedding that way is close to exact.

The notebook does not cover runs 82 to 93, so I don't know what helper functions and variables exist now. Next I will list the kernel's user-defined variables and functions and check the size of ARCHITECTURE.md. That tells me what to re-read before continuing. If a step function such as "e_route" exists and is verified, I move on to the value head and to local rollouts. If not, I rebuild the missing parts.

```python
import types, inspect, os, numpy as np
skip = set(dir(__builtins__)) | {'In','Out','exit','quit','get_ipython'}
rows=[]
for k,v in list(globals().items()):
    if k.startswith('_') or k in skip: continue
    if isinstance(v, types.ModuleType): continue
    t=type(v).__name__
    info=''
    if hasattr(v,'shape'): info=str(tuple(v.shape))
    elif isinstance(v,(list,tuple,dict,set)): info='len=%d'%len(v)
    elif callable(v):
        try: info=str(inspect.signature(v))
        except Exception: info='?'
    rows.append((k,t,info))
for r in rows: print(r[0], r[1], r[2][:90])
print(os.path.getsize('/model/ARCHITECTURE.md'), os.path.getsize('/work/notes.md'))
```