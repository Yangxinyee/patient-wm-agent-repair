Run 142 compared choices for how the insulin signal gets written into the model. A stronger off-target penalty (woff) shrinks the stray write on non-drug bundles: the p99 falls from about 1065 to about 5. The cost is a noisier read of the insulin rate: the residual sd rises from 0.13 to 0.46, where the dose deviation has sd about 0.59. The drug/non-drug gate works well. On validation, drug bundles score at least 0.878 and non-drug bundles at most −0.527, so writes can be limited to drug bundles.

My conclusion is to use the gate to block off-target writes rather than a heavy penalty. A light penalty (woff around 1e-4 to 1e-3) would then keep the insulin read accurate.

My notes stop at run 120, so the details of runs 121–142 are only in kernel variables. Before building anything I will list what the kernel holds (functions with signatures, arrays with shapes, dicts with keys). I will also log run 142 in the notebook. If the list shows the write-fit and gate objects from run 142, I can continue from them; if not, I will rebuild them from the validated replica functions.

```python
import types, inspect, numpy as np, torch, os
g = dict(globals())
skip = {'In','Out','get_ipython','exit','quit'}
fns, arrs, scal, conts, oth = [], [], [], [], []
for k, v in g.items():
    if k.startswith('_') or k in skip or isinstance(v, types.ModuleType): continue
    if isinstance(v, types.FunctionType):
        try: sig = str(inspect.signature(v))
        except Exception: sig = '(?)'
        fns.append(f'{k}{sig}')
    elif isinstance(v, np.ndarray): arrs.append(f'{k}:{v.shape}')
    elif isinstance(v, torch.Tensor): arrs.append(f'{k}:T{tuple(v.shape)}')
    elif isinstance(v, (int, float, str, bool)): scal.append(f'{k}={repr(v)[:40]}')
    elif isinstance(v, (list, tuple, dict, set)):
        extra = ''
        if isinstance(v, dict) and len(v) <= 40: extra = ' keys=' + ','.join(map(str, list(v.keys())[:40]))
        conts.append(f'{k}:{type(v).__name__}[{len(v)}]{extra[:300]}')
    else: oth.append(f'{k}:{type(v).__name__}')
print('FUNCTIONS', len(fns)); print('\n'.join(fns))
print('\nARRAYS', len(arrs)); print('; '.join(arrs))
print('\nSCALARS', len(scal)); print('; '.join(scal))
print('\nCONTAINERS', len(conts)); print('\n'.join(conts))
print('\nOTHER', '; '.join(oth))
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 5 (run 143)\n- Run142: write fit with off-target penalty woff: woff 1e-4 ins resid .255 offtarget p99 72; 1e-3 .34/29; 1e-2 .40/10; 0.1 .46/5. "
            "Drug/non-drug GATE separates cleanly (VALID drug min .878, nondrug max -.527) -> use gate to suppress off-target writes, keep woff small.\n")
print(os.path.getsize('/work/notes.md'))
```