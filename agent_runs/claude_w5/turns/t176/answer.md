Restarted after several of my replies were blocked. I'll keep outputs short and strictly about the model's weights and the simulated records.

**Where things stand:**
- I have an exact local copy of the core step (`cstep`, `enc2`) and of the head.
- The last run failed only because a new helper used a weight name that does not exist.
- Instead of writing new code, I will check which kernel objects and weight-name patterns exist, and reuse the verified functions.

**Working plan (being written to the notebook):**
- Make the model carry the drug's effect in its state.
- At each drug bundle, add to the core state the change that a lower (insulin) or higher (norepinephrine) last reading would have caused.
- Scale that change by the dose minus the policy-expected dose, so factual predictions stay centred.
- Set the scale against the effect sizes I estimated from the records.

**What would confirm the plan is workable:** the kernel still has `cstep` and the bundle data, and the weight names show where `b_in` and the prelude attention live.

```python
import numpy as np, json
g=globals()
fn=[k for k in g if not k.startswith('_') and callable(g[k]) and not isinstance(g[k],type)]
var=[k for k in g if not k.startswith('_') and not callable(g[k]) and k not in ('In','Out')]
print('fns:',fn[:60])
print('vars:',[k for k in var][:60])
pre=sorted(set('.'.join(n.split('.')[:3]) for n in NAMES))
print(len(pre)); print(pre[:70])
with open('/work/notes.md','a') as f:
    f.write("\n## Restart 4 (run 157)\n- PLAN: belief-injection edit. At drug bundles add to core state kappa*(rate-c_hat)*v, v = state change per unit change of last reading (finite diff, local core). Calibrate kappa with dose_edit 'both' on tuning sel A vs g-est truth; check factual NLL.\n")
```