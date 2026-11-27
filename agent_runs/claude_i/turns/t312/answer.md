The kernel already holds a lot from earlier sessions:
- the z-posterior network: one hidden layer of 192 units, input [ctx, e, u] of size 576, output 64;
- its 50 dead units (`dead`);
- `Jtok` (128×192×4), apparently a Jacobian of 4 predictions with respect to the 192-dim z token on 128 bundles. I probably computed it in the session whose outputs were blocked;
- `core_step`, which reads weights through a global `WT`.

**Next:** check what `WT` is, so I can swap in a new z_in column for local tests. I will also find out what `Jtok` and `G_tok_A` contain (which outputs and bundles), and list the remaining arrays that were cut off.

**What would confirm the plan:** `WT` is a dictionary I can edit, and `Jtok` shows a consistent direction that moves the glucose prediction. If the Jacobian directions vary wildly between bundles, a single fixed z_in column would not work and I need a different route.

```python
import numpy as np, re, types
print('WT', type(WT).__name__, len(WT) if hasattr(WT,'__len__') else '')
if isinstance(WT, dict): print([k for k in WT if ('z_in' in k or 'z_post' in k)][:6], type(next(iter(WT.values()))).__name__)
d = np.load('/work/Jtok.npz'); print('Jtok.npz', {k: d[k].shape for k in d.files})
print('G_tok_A', np.load('/work/G_tok_A.npy').shape)
for fp in ['jac2.py','sim.py']:
    s = open('/work/'+fp).read().splitlines()
    print(fp, [l.strip()[:100] for l in s if re.search(r'def |save|return', l)][:10])
print('Jtok |mean| per out', np.abs(Jtok).mean((0,1)).round(4), 'base4 mean', base4.mean(0).round(2))
Jm_ = Jtok.mean(0); rms = np.sqrt((Jtok**2).sum(1).mean(0))
print('mean-J norm', np.linalg.norm(Jm_,axis=0).round(4), 'rms norm', rms.round(4))
print('cos(meanJ out0..3)', np.round([[float(Jm_[:,i]@Jm_[:,j]/np.linalg.norm(Jm_[:,i])/np.linalg.norm(Jm_[:,j])) for j in range(4)] for i in range(4)],2).tolist())
print('dead[:8]', dead[:8], 'deadH', deadH)
g = dict(globals())
arr = sorted((k, tuple(v.shape)) for k,v in g.items() if isinstance(v, np.ndarray) and not k.startswith('_'))
print('ARR2:', '; '.join(f'{k}{s}' for k,s in arr if k >= 'dtb')[:1500])
lst = sorted((k, len(v)) for k,v in g.items() if isinstance(v,(list,tuple)) and not k.startswith('_') and k not in ('In','Out'))
print('LISTS:', lst[:25])
```