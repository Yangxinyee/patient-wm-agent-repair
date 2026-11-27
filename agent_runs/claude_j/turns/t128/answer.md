The last few replies failed only because their output was too long. I'll keep every printout to short summaries from now on. Before building the planned edit (a dose-correction readout in the value head), I need to know what the kernel still holds: the replica functions, weight dictionaries, bundles and states. If the replica (`step`, `build_bundles`) and head code are still there, I can reuse them. If not, I'll rebuild from `/work/*.py`.

```python
import types
names=[k for k,v in globals().items() if not k.startswith('_') and not isinstance(v,types.ModuleType) and k not in ('In','Out','exit','quit','get_ipython')]
fn=[k for k in names if callable(globals()[k]) and not isinstance(globals()[k],type)]
arr={k:tuple(getattr(globals()[k],'shape')) for k in names if hasattr(globals()[k],'shape') and k not in fn}
others=[k for k in names if k not in fn and k not in arr]
print(len(names)); print('FNS:',' '.join(fn)[:1200]); print('ARR:',str(arr)[:1200])
print('OTHER:',' '.join(f"{k}:{type(globals()[k]).__name__}" for k in others)[:1200])
import os; print({f:os.path.getsize('/work/'+f) for f in os.listdir('/work') if f.endswith(('.py','.md','.json'))})
```