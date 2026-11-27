Picking up after the restart. The last output (run 133) compared variants of the atom-embedding composition. "none" gave an error of 2.88 and "unit" gave 2.45. The notebook does not record what that error measured, so I can't tell yet whether there was a remaining mismatch. Before going on I'll re-read the whole notebook and impl.py and list the kernel variables to rebuild context. The notebook says the encoder reproduces the z posterior exactly. If the run-133 comparison turns out to concern a later stage (core step or heads), that tells me which piece of the local re-implementation still needs fixing.

```python
import os, json, types
print(open('/work/notes.md').read()[:12000])
print("=====IMPL=====")
src=open('/work/impl.py').read()
print(len(src)); print(src[:9000])
print("=====VARS=====")
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v,(types.ModuleType,types.FunctionType,type)): continue
    try:
        sh = getattr(v,'shape',None); 
        print(k, type(v).__name__, sh if sh is not None else (len(v) if hasattr(v,'__len__') else v) )
    except Exception as e: print(k,'?')
print([k for k,v in globals().items() if isinstance(v,types.FunctionType)])
```