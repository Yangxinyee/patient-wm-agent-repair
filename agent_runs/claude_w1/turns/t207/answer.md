The notebook start confirms where the target effects came from. They come from exogenous dose noise in the training records (multiplicative policy noise uncorrelated with state) and are cross-checked by a regression discontinuity. The evidence: insulin lowers finger glucose over about 12 hours (cumulative about -13.7 per unit/h), norepinephrine raises arterial MAP (cumulative about +9.3 per mcg/kg/min), and there are no cross effects. On setA that gives targets of -14.17 / +7.09 / +4.82 / -2.41, against the original model's -1.45 / +0.82 / +0.15 / -0.41.

Next I need to see what the kernel still holds (decoders, head replica, sweep files) before I build the value-head edit. I will print only names, types and shapes.

```python
import numpy as np, types
out=[]
for k,v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    if isinstance(v, np.ndarray): out.append(f"{k}:arr{v.shape}")
    elif isinstance(v, dict): out.append(f"{k}:dict{len(v)}")
    elif isinstance(v, (list,tuple)): out.append(f"{k}:list{len(v)}")
    elif callable(v) and not isinstance(v,type): out.append(f"{k}()")
print(len(out)); print(' '.join(out)[:3500])
for f in ['sweep_h2.npz','sweep_dwc.npy']:
    try:
        d=np.load('/work/'+f, allow_pickle=True)
        print(f, {kk:d[kk].shape for kk in d.files} if hasattr(d,'files') else d.shape)
    except Exception as e: print(f, e)
```