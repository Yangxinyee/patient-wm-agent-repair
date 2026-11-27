The notebook confirms the timing. Readings fall on integer hours. Each infusion starts at h+0.083 h and runs one hour, with the rate chosen from the latest glucose. So a reading at hour t is predicted from the state after the drug bundle at t-1+0.083, and "lag k" means the infusion started k hours before. Next I list the kernel's variables (dataframes, weight dicts, selections), so I can reuse the loaded data and the weight dict `W` instead of rebuilding them.

```python
import types
rows=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v,(types.ModuleType,types.FunctionType,type)): continue
    t=type(v).__name__
    if hasattr(v,'shape'): d=str(tuple(v.shape))
    elif isinstance(v,(list,dict,tuple,set)): d='len %d'%len(v)
    else: d=str(v)[:40]
    rows.append((k,t,d))
for r in rows: print('%-18s %-14s %s'%r)
print([k for k,v in globals().items() if isinstance(v,types.FunctionType)])
```