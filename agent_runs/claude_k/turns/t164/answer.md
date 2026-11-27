I'm restarting with little context, so first I'll re-read the architecture file and list what the kernel holds. I'll also check whether the saved writer edit (dose to z dims 18 and 28) does what I meant:
- On drug bundles the shift should land only in z18 and z28.
- It should be zero elsewhere.
- I also want to see how far it spreads into the later state.

If the shift shows up mainly in z18/z28 on drug bundles and nowhere else, the writer works. Large shifts elsewhere, or non-zero shifts on non-drug bundles, would mean my reconstruction of the edit was wrong.

```python
import numpy as np, json, os, types
print(open('/model/ARCHITECTURE.md').read())
gl = {}
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    if callable(v) and not hasattr(v,'shape'): gl[k]='fn'
    elif hasattr(v,'shape'): gl[k]=str(tuple(v.shape))
    elif isinstance(v,(list,dict,tuple)): gl[k]=type(v).__name__+str(len(v))
    elif isinstance(v,(int,float,str)): gl[k]=repr(v)[:30]
print(gl)
d8o=np.load('/work/fwd_8/states_original.npz'); d8e=np.load('/work/fwd_8/states_edited.npz')
X8o=d8o['x']; X8e=d8e['x']; print(list(d8o.keys()), X8o.shape)
print('meta equal', all((d8o[k]==d8e[k]).all() for k in ['stay_id','hour','has_insulin','has_norepinephrine','n_atoms']))
dz=X8e[:,960:992]-X8o[:,960:992]; hi=d8o['has_insulin'].astype(bool); hn=d8o['has_norepinephrine'].astype(bool)
for nm,m in [('ins',hi),('ne',hn),('none',~hi&~hn)]:
    print(nm, m.sum(), 'mean|dz| per dim top:', np.round(np.abs(dz[m]).mean(0),3)[[18,28]], 'max other', np.round(np.delete(np.abs(dz[m]).mean(0),[18,28]).max(),4))
dout=np.linalg.norm(X8e[:,:192]-X8o[:,:192],axis=1); dh=np.linalg.norm(X8e[:,192:960]-X8o[:,192:960],axis=1)
print('norm out orig', np.linalg.norm(X8o[:,:192],axis=1).mean(), 'dout mean', dout.mean(), 'dh mean', dh.mean(), 'h norm', np.linalg.norm(X8o[:,192:960],axis=1).mean())
```