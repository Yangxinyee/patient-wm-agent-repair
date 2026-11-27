The last output was blocked, so I don't know what runs 234 to 236 produced. Before spending a forward pass, I'll take a short inventory: which tensors are edited and by how much, and which key variables survive in the kernel (lamA/lamB, the decoders, the impulse responses). This confirms whether writer v2 is the edit currently saved. If only rows/cols 12, 88 and 105 of z_post differ from the original, my notes are accurate.

```python
import numpy as np
def A(x): 
    try: return x.detach().cpu().numpy()
    except Exception: return np.asarray(x)
print(edited())
for n in ['z_post.0.weight','z_post.0.bias','z_post.2.weight','z_post.2.bias']:
    a=A(load(n)); b=A(load_original(n)); d=a-b
    nz=np.argwhere(np.abs(d)>0)
    rows=sorted(set(nz[:,0].tolist())) if len(nz) else []
    cols=sorted(set(nz[:,1].tolist())) if (len(nz) and d.ndim==2) else []
    print(n, a.shape, 'n', len(nz), 'rows', rows[:8], 'cols', cols[:8], 'max', round(float(np.abs(d).max()),3))
g=[k for k in globals().keys() if not k.startswith('_') and not callable(globals()[k])]
print(len(g))
print(' '.join([k for k in g if len(k)<=10][-170:]))
for k in ['lamA','lamB','j0','jA','jB']:
    if k in globals(): print(k, globals()[k])
```