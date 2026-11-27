The config gives 4 attention heads of size 48, 1 prelude layer, 2 recurrent layers, 1 coda layer and a memory of 128 entries. There are no token-type embeddings, and z enters the prelude only through z_in. So a local core step can be written as standard pre-norm attention and FFN layers. The open details are masking of absent tokens, the age keys of the cross-attention and memory timing, and I can settle those by matching the model's real states.

Next I will check what is already in the kernel and the layout of the saved model states (fwd_2), so I can line up my bundles with the model's.

- **Confirms:** the npz holds per-bundle out and h, plus stay, hour and time indices, so I can validate a local step.
- **Refutes:** the bundles cannot be aligned. Then I would need another states pass.

```python
import numpy as np, types
gl={k:v for k,v in list(globals().items()) if not k.startswith('_') and not callable(v) and not isinstance(v, types.ModuleType)}
for k,v in gl.items():
    if isinstance(v,(np.ndarray,)): print(k,'arr',v.shape,v.dtype)
    elif hasattr(v,'shape'): print(k,type(v).__name__,getattr(v,'shape',None))
    elif isinstance(v,(dict,list,tuple)): print(k,type(v).__name__,len(v))
    elif isinstance(v,(int,float,str)) and len(str(v))<60: print(k,'=',v)
import os
for d in ['fwd_1','fwd_2','fwd_3']: print(d, os.listdir('/work/'+d))
z=np.load('/work/fwd_2/states_original.npz', allow_pickle=True)
for k in z.files: print(k, z[k].shape, z[k].dtype, z[k][:3] if z[k].ndim==1 else '')
```