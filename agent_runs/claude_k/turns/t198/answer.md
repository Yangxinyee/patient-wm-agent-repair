Run 183 was only a probe of helper functions in the kernel. It showed two problems:
- The measurement-atom rebuilder `meas_atoms` depends on a global `rd`, and `rd` has been overwritten with a string.
- A later line in that run failed on an array that is 1-D, not 2-D as I assumed.

Runs 168–183 were never logged in the notebook, and my reply after run 183 was withheld. So my picture of the kernel state is incomplete.

Before more pipeline work I need an accurate inventory: which helper functions, arrays and fitted objects exist, with their shapes. I also need to re-read the start of the notebook (data layout and early findings), because only its tail was shown.

**Current plan.** The edit should feed the dose *innovations* (dose minus the policy-expected dose) into the model through a clean channel. That way the model's dependence on dose matches the effect estimated from the records, while factual predictions improve or stay the same. The present writer edit (z_post units 55 and 109) still has the empty-action bug and must be rebuilt.

**What this run should settle.** If the inventory shows the reconstruction pieces are intact (`u_rec`, `datom`, `view_from`, the Q99/WC99 arrays, the p_ins/p_ne decoders, the fwd8 states), I can fix the writer quickly. If they are gone, I will rebuild the minimum needed.

```python
import types, inspect, numpy as np
txt = open('/work/notes.md').read()
print(len(txt)); print(txt[:5200])
print('=====')
g = dict(globals())
funcs, arrs, others = [], [], []
for k, v in g.items():
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    if isinstance(v, types.FunctionType):
        try: sig = str(inspect.signature(v))
        except Exception: sig = '(?)'
        funcs.append(k + sig[:50])
    elif isinstance(v, np.ndarray):
        arrs.append(f"{k}{list(v.shape)}")
    elif hasattr(v, 'shape') and not isinstance(v, type):
        try: arrs.append(f"{k}:{type(v).__name__}{list(v.shape)}")
        except Exception: others.append(k)
    elif isinstance(v, (bool, int, float, str)):
        others.append(f"{k}={repr(v)[:25]}")
    elif isinstance(v, (list, tuple, dict, set)):
        others.append(f"{k}:{type(v).__name__}{len(v)}")
    else:
        others.append(f"{k}:{type(v).__name__}")
print('FUNCS', len(funcs)); print('; '.join(funcs))
print('ARRS', len(arrs)); print('; '.join(arrs))
print('OTHERS', len(others)); print('; '.join(others))
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 4 (after run 183)\n- runs 168-183 not logged; run 184 = inventory of kernel state + notes head.\n"
            "- Design principle: correction = sum_l kappa_l * dose_innovation_{t+1-l} (innovation = dose - policy-expected dose given readings/prev dose),\n"
            "  kappa from regressing model factual error on innovations; written via clean z channel; then rollout handles propagation. Verify with dose_edit.\n")
```