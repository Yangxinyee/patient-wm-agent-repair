The helpers I need already exist:
- `u_for(k, fI, fN)` builds the drug token under a dose multiplier.
- `next_pred(k, ztok_add, codes)` gives the next-bundle prediction with an added z-token vector.
- `jac_at` / `best_dir` computed and used the token Jacobians in Jtok.
- `run_stay_z` runs a stay with a z addition.

WT and dts are float32, so the earlier failure came from float64 inputs (probably the u from u_for), which I'll cast.

Next I'll list the kernel's data variables (names and shapes only). I need to find the per-bundle encodings (e, u, ctx), the bundle metadata (stay, hour, insulin rate, NE rate, decision glucose) and whatever produced "ins bundles 1362 usable 1301", so the decoders and gate can be fit without rebuilding the data.

```python
import pandas as pd
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or callable(v) or k in ('In','Out'): continue
    if isinstance(v,pd.DataFrame): out.append(f"{k}[DF{v.shape}:{','.join(map(str,v.columns[:6]))}]")
    elif hasattr(v,'shape') and len(getattr(v,'shape',()))>0: out.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v,(list,dict,tuple)) and len(v)>0: out.append(f"{k}<{type(v).__name__}{len(v)}>")
s=' '.join(out); print(len(out)); print(s[:3500])
```