There is no input history (`In` is None), so the only way to recover runs 213–249 is from the kernel variables. Before anything else I'll inspect:
- the defined functions, which name the pipeline steps;
- what SOLU3 is;
- variables that look like recent diagnostics.

I'll also compare fwd_5 directly: the tool's edited-minus-original states against my local edited-minus-original states on the same 13 tuning stays. That tells me whether the tool runs the edit the way I simulate it in factual mode.

- If the differences match, the edit works teacher-forced and the rollout failure is about dynamics: the decoder not responding to the tool's dose edit, or the injected shift washing out.
- If they don't match, my local model of the edited network is wrong somewhere.

```python
import types, numpy as np
G_ = globals()
fn = sorted(k for k in G_ if isinstance(G_[k], types.FunctionType))
print('functions', len(fn)); print(' '.join(fn))
for nm in ['SOLU3','SOL','SOLP','DU3','EE_cache','KE','Kem']:
    if nm in G_:
        v = G_[nm]
        if isinstance(v, dict): print(nm, 'dict', list(v.keys())[:12])
        elif isinstance(v, (list, tuple)): print(nm, type(v).__name__, len(v), type(v[0]).__name__ if len(v) else '')
        else: print(nm, type(v).__name__, getattr(v,'shape',None), (float(np.abs(v).max()) if hasattr(v,'shape') else v) if not callable(v) else '')
low = sorted(k for k in G_ if k[0].islower() and not isinstance(G_[k], (types.FunctionType, types.ModuleType, type)))
print('lowercase non-fn vars', len(low)); print(' '.join(low)[:4000])
```