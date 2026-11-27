Run 144 listed the newest helpers: `build_ds`, `tf_eval`, `tf_loop`, `prelude_parts`, `wfit_eval`, `A3_of`, `flag`, `report`, `kill_dims`. Its arrays listing stopped at about 3,500 characters. The variables from runs 121–142 (the write targets, gate, and prelude features) were created most recently, so they sit at the end of the list and were cut off.

Next I'll print only the last ~150 arrays, ~50 containers and ~60 scalars. I'll also print the bytecode names that `wfit_eval`, `A3_of`, `prelude_parts` and `tf_loop` reference (`co_names`, `co_varnames`) to recover how they work, since their source text is gone. I expect to find the feature matrix XX, the insulin targets y and the gate weights from run 142. If they are missing, I'll rebuild them with `A3_of`/`prelude_parts` on a train ds.

```python
import types, numpy as np, torch
g = dict(globals())
skip = {'In','Out','get_ipython','exit','quit'}
arrs, scal, conts = [], [], []
for k, v in g.items():
    if k.startswith('_') or k in skip or isinstance(v, types.ModuleType) or isinstance(v, types.FunctionType): continue
    if isinstance(v, np.ndarray): arrs.append(f'{k}:{v.shape}')
    elif isinstance(v, torch.Tensor): arrs.append(f'{k}:T{tuple(v.shape)}')
    elif isinstance(v, (int, float, str, bool)): scal.append(f'{k}={repr(v)[:30]}')
    elif isinstance(v, (list, tuple, dict, set)):
        extra = ''
        if isinstance(v, dict) and len(v) <= 20: extra = ' keys=' + ','.join(map(str, list(v.keys())[:20]))
        conts.append(f'{k}:{type(v).__name__}[{len(v)}]{extra[:150]}')
print('LAST ARRAYS:', '; '.join(arrs[-150:])[:3800])
print('\nLAST SCALARS:', '; '.join(scal[-60:])[:1500])
print('\nLAST CONTAINERS:', '\n'.join(conts[-45:])[:2200])
for fnm in ['wfit_eval','A3_of','prelude_parts','tf_loop','report','flag','build_ds','tf_eval']:
    co = g[fnm].__code__
    print(f'\n{fnm}: args={co.co_varnames[:co.co_argcount]} names={co.co_names[:40]} locals={co.co_varnames[co.co_argcount:][:40]}')
```