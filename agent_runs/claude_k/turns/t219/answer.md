The architecture doc is short (5,234 characters, 92 lines). The sections that matter are "State and one step per bundle", "Heads" and "How a dose edit is read", lines 30–92. My inventory code tripped over the numpy module itself, which has `shape` and `dtype` attributes. I'll filter by real array types instead.

Next I'll print lines 30–92 of the doc in full, about 3,500 characters, so I can see exactly how u enters the state and how state info decays into the heads. The hypothesis to check is that there is a recurrent slot path with a per-dimension time decay, which the heads read directly. A path like that could carry a dose impulse response; mixing only through z would not.

```python
import numpy as np, torch, types
print('\n'.join(L[30:92]))
inv=[]
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray, torch.Tensor)): inv.append(f"{k}{tuple(v.shape)}")
    elif isinstance(v, types.FunctionType): inv.append(f"{k}()")
print(len(inv)); print(' '.join(inv)[:2500])
```