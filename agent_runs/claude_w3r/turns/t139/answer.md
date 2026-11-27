Two restarts in a row left the notebook ending at the Session 5 plan. Runs 118–127 have no entries, so I don't know what they found. The kernel still holds my variables, possibly including the local core-step code, which I never saved to a file.

Next step: list the functions defined in the kernel and save their source to /work/impl3.py. I'll also print the shapes of the unrecorded variables (E160, J, Q, A, XW, QW, usage, W2 and others) so I can rebuild what runs 118–127 did.

- **Confirming outcome:** the core-step functions are still in the kernel and their source can be recovered. I can then go straight to designing the dose-memory channel.
- **Refuting outcome:** they are gone or the source can't be read. I would then rebuild the core step from my notes before going on.

```python
import inspect, types, numpy as np, torch
g = globals()
fns = {k: v for k, v in g.items() if isinstance(v, types.FunctionType) and not k.startswith('_')}
print('functions:', sorted(fns))
ok, bad = [], []
with open('/work/impl3.py', 'w') as fh:
    for k, v in fns.items():
        try:
            fh.write(inspect.getsource(v) + '\n\n'); ok.append(k)
        except Exception:
            bad.append(k)
print('source saved:', ok)
for k in bad:
    co = fns[k].__code__
    print('NO SRC', k, 'args', co.co_varnames[:co.co_argcount], 'names', co.co_names[:25])
for k in ['E160','J','Q','A','view','XW','QW','usage','W2','MT','mx','D','Rv','R','pre','pre2','sa','mk','act','grp','sel','Xc','lr','lr2','FA','meta','PA','Gf','Ins','glc','fresh','PREV','PREVt','LASTREG','HD','FH','DH']:
    v = g.get(k)
    if v is None: print(k, 'None'); continue
    if isinstance(v, (np.ndarray, torch.Tensor)): print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (list, tuple, dict)): print(k, type(v).__name__, len(v), (list(v.keys())[:8] if isinstance(v, dict) else ''))
    else: print(k, type(v).__name__, str(v)[:100])
```