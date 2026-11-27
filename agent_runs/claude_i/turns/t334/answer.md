That run was wasted on a bug in my listing code, not in the model. One global is a numpy function whose `.shape` attribute is itself a function, so `len()` failed. I'll redo the same listing with each item wrapped in try/except, since I still need the names of the per-bundle encodings and metadata before fitting the dose decoders. 23 runs remain.

```python
import pandas as pd
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or callable(v) or k in ('In','Out'): continue
    try:
        if isinstance(v,pd.DataFrame): out.append(f"{k}[DF{v.shape}:{','.join(map(str,v.columns[:6]))}]")
        elif isinstance(getattr(v,'shape',None),tuple) and len(v.shape)>0: out.append(f"{k}{tuple(v.shape)}")
        elif isinstance(v,(list,dict,tuple)) and len(v)>0: out.append(f"{k}<{type(v).__name__}{len(v)}>")
    except Exception: pass
s=' '.join(out); print(len(out)); print(s[:3500])
```