The architecture file confirms how the dose reaches the core. It enters only as u_k, which feeds `z_post([ctx, e, u])` and is also a token in the prelude. So the dose memory has to live in h. Before planning an edit around the z posterior I need to know why my z_post replica is still off. The candidates are the input order, the split of mean and log-sd in the output, or a different gap module. I'll keep this output short.

```python
for n in NAMES:
    if n.startswith('z_') or 'gap' in n or n.startswith('core.out_norm') or n.startswith('z_in'):
        print(n, SHAPES[n])
s = open('/work/setup.py').read(); print(len(s)); print(s[:1500])
import numpy as np, torch
for k,v in list(globals().items()):
    if k.startswith('_'): continue
    if isinstance(v,(np.ndarray,torch.Tensor)) and v.ndim>=1 and v.size if isinstance(v,np.ndarray) else False:
        pass
arrs = [(k, tuple(v.shape)) for k,v in globals().items() if not k.startswith('_') and isinstance(v,(np.ndarray,torch.Tensor))]
print(arrs[:80])
```