The inventory was cut off. One function, `tf_states`, has a huge default argument, and printing its signature used up the 10,000-character limit. The functions listed before it are as expected: the replica pieces (`core_step_b`, `head_value`, `tf_stay`, `rollout_stay`, `rollout_gen`, `step_b`, `MH`/`LN`/`FF`) and helpers like `rates_by_bundle` and `stay_inputs`. The arrays, scalars and dicts never printed.

I'll repeat the inventory with each function signature capped at 100 characters, then print the arrays, scalars and containers. That should show the run-142 write and gate objects and the current parameter dict. If they appear (a P dict, write matrices, gate weights), I'll build the insulin edit on them. If not, I'll rebuild them from the validated replica functions.

```python
import types, inspect, numpy as np, torch
g = dict(globals())
skip = {'In','Out','get_ipython','exit','quit'}
fns, arrs, scal, conts, oth = [], [], [], [], []
for k, v in g.items():
    if k.startswith('_') or k in skip or isinstance(v, types.ModuleType): continue
    if isinstance(v, types.FunctionType):
        try: sig = str(inspect.signature(v))
        except Exception: sig = '(?)'
        fns.append(f'{k}{sig[:100]}')
    elif isinstance(v, np.ndarray): arrs.append(f'{k}:{v.shape}')
    elif isinstance(v, torch.Tensor): arrs.append(f'{k}:T{tuple(v.shape)}')
    elif isinstance(v, (int, float, str, bool)): scal.append(f'{k}={repr(v)[:40]}')
    elif isinstance(v, (list, tuple, dict, set)):
        extra = ''
        if isinstance(v, dict) and len(v) <= 40: extra = ' keys=' + ','.join(map(str, list(v.keys())[:40]))
        conts.append(f'{k}:{type(v).__name__}[{len(v)}]{extra[:200]}')
    else: oth.append(f'{k}:{type(v).__name__}')
i0 = [i for i,f in enumerate(fns) if f.startswith('tf_states')]
print('FUNCTIONS after tf_states:'); print('\n'.join(fns[i0[0]:] if i0 else fns))
print('\nARRAYS', len(arrs)); print('; '.join(arrs)[:3500])
print('\nSCALARS', len(scal)); print('; '.join(scal)[:1500])
print('\nCONTAINERS', len(conts)); print('\n'.join(conts)[:2500])
print('\nOTHER', '; '.join(oth)[:600])
```