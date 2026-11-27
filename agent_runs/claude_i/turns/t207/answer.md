The shapes confirm the injection route:
- `z_post` is one hidden layer of 192 GELU units (input [ctx, e, u] of size 576), followed by a linear layer giving a 64-wide output (mean, then log-sd). Repurposing the dead hidden units is therefore a clean, local change.
- `z_in` maps the 32-dimensional z into the prelude, so a new direction added to z's mean reaches the state update.
- The value head is a two-layer MLP over `with_code`.
- `last_numeric` has 8 input columns, not the 4 the document names. That probably explains why my local glucose-head copy disagrees with the tool, so I will calibrate glucose against the tool, not my copy.

Next I need an inventory of the arrays still in the kernel: the states from the third forward pass, contexts, encodings and bundle metadata. Then I can find the dead units and compute how a change in z at insulin bundles moves the next glucose prediction.

If the inventory includes per-bundle metadata (stay, hour, drug flags, gaps) and the recomputed contexts, I can run that analysis locally without spending forward passes. If not, I will rebuild them.

```python
import numpy as np, torch
rows=[]
for k,v in list(globals().items()):
    if k.startswith('_') or callable(v) or k in ('In','Out'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)): rows.append((k,type(v).__name__,tuple(v.shape)))
    elif isinstance(v,(pd.DataFrame,)): rows.append((k,'DF',v.shape))
    elif isinstance(v,(dict,list,tuple)) and k not in ('NAMES','SHAPES','VOCAB'): rows.append((k,type(v).__name__,len(v)))
    elif isinstance(v,(int,float,str)) and len(str(v))<40: rows.append((k,type(v).__name__,v))
print(len(rows))
print('; '.join(f"{a}:{b}{c}" for a,b,c in sorted(rows)))
```