The last run loaded the original model's teacher-forced predictions on the 100 tuning stays (sel100). The value NLLs per event match the earlier fwd_2 summary: finger 1.552, lab 1.240, arterial 1.352, cuff 1.692. The residual means are small: glucose +2.7 and +3.8 mg/dL, MAP −0.9 and −1.5 mmHg. So that baseline is consistent.

No tensors are edited right now. The only untested candidate is `build_v3`, which carries the dose through the head-1 value path. Head 1 reads the action token with weight about 1.0 at every bundle, so this path is genuinely used, unlike the z token that made v1 fail.

I'll stay strictly within the rules: only the listed files, no gradients, and an edit that represents the drug effect the same way in factual and rollout use, with no mode detection.

Next I'll take an inventory of what the kernel still holds: the functions with their arguments and the arrays with their shapes. I'll also look at the global names that `build_v3` uses, so I can rebuild the edit without guessing.

- **Confirms the plan:** `build_v3`, the readouts `qd`, the directions (aI3, aN3, aP3, dI, dN) and the replica step functions all exist. Then I check the edit numerically on the CPU and spend one forward pass on it.
- **Refutes it:** they are missing or inconsistent. Then I rebuild the head-1 carrier from the documented pieces.

```python
import types, dis, io, contextlib
import numpy as np, torch
rows=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    if isinstance(v, np.ndarray): rows.append(('nd',k,str(v.shape)))
    elif isinstance(v, torch.Tensor): rows.append(('pt',k,str(tuple(v.shape))))
    elif isinstance(v, types.FunctionType):
        c=v.__code__; rows.append(('fn',k,str(c.co_varnames[:c.co_argcount])))
    elif isinstance(v,(dict,list,tuple,set)):
        extra=''
        if isinstance(v,dict): extra=str(list(v.keys())[:8])
        rows.append((type(v).__name__,k,f'len={len(v)} {extra}'[:110]))
    elif isinstance(v,(int,float,str,bool,np.floating,np.integer)): rows.append(('sc',k,str(v)[:60]))
    elif hasattr(v,'shape'): rows.append((type(v).__name__,k,str(getattr(v,'shape',''))))
    else: rows.append((type(v).__name__,k,''))
rows.sort()
for t,k,s in rows:
    if t in ('fn',): print(f'{t:5s} {k:22s} {s}')
print('-----')
for t,k,s in rows:
    if t not in ('fn',): print(f'{t:9s} {k:20s} {s}')
# global names used inside build_v3
if 'build_v3' in globals():
    c=build_v3.__code__
    print('build_v3 globals used:', c.co_names)
    print('build_v3 locals:', c.co_varnames)
    print('consts:', [x for x in c.co_consts if not isinstance(x, types.CodeType)][:40])
import os
print(os.path.getsize('/work/notes.md'), 'bytes notes;', os.path.getsize('/model/ARCHITECTURE.md'),'bytes arch')
```