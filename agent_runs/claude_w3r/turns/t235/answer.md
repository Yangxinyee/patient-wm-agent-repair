The last check passed. My local copy of the prelude attention (`patt`) and head output (`HO`) matches the model to about 1e-7, so that code can be trusted. The no-action `u` token is the zero vector, and norm1 maps it to the bias, which is small (max 0.08). Runs 208–222 were never written into the notebook, so I don't know what the newest kernel objects are. Next I'll recover them: list the kernel globals in the order they were defined (the newest show my recent work), the functions in the `/work/impl*.py` files, and the selection files. I'll also add a session header to the notebook. If this shows partly built channel-stage functions and tensors, I'll continue from them. If not, I'll rebuild stage 1 (prelude rank-1 write) from the validated `patt`/`HO` code.

```python
import os, types, re, json
import numpy as np
G = [(k,v) for k,v in list(globals().items()) if not k.startswith('_')]
def desc(k,v):
    if isinstance(v, types.ModuleType): return None
    if isinstance(v, np.ndarray): return f"{k}{list(v.shape)}"
    if hasattr(v,'shape') and 'torch' in str(type(v)): return f"{k}T{list(v.shape)}"
    if isinstance(v, types.FunctionType): return f"{k}()"
    if isinstance(v,(int,float,np.floating)): return f"{k}={v:.4g}" if isinstance(v,(float,np.floating)) else f"{k}={v}"
    if isinstance(v,(dict,list,tuple)): return f"{k}:{type(v).__name__}{len(v)}"
    return f"{k}:{type(v).__name__}"
D=[d for d in (desc(k,v) for k,v in G) if d]
print(len(D), "globals. LAST 110 (chronological):")
print(' '.join(D[-110:]))
for f in sorted(os.listdir('/work')):
    p='/work/'+f
    if f.endswith('.py'):
        s=open(p).read(); print(f, len(s), re.findall(r'^def (\w+)', s, re.M))
    elif f.endswith('.json'):
        s=json.load(open(p)); print(f, type(s).__name__, (len(s) if hasattr(s,'__len__') else s), str(s)[:120])
with open('/work/notes.md','a') as fh:
    fh.write("\n## Session 9 (restart 9) - runs 208-222 not noted; run 222: local prelude attn (patt) & head-output (HO) match model to 1e-7; u0=0 -> norm1(u0)=bias (max .08).\n")
```