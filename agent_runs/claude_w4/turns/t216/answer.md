Run 189 tested the drug-rate readouts fI (insulin) and fN (norepinephrine) on modified doses. Both track the rate closely:
- **Bias:** under 0.025 in most dose variants (worst +0.046, at norepinephrine x3).
- **Insulin rms:** 0.06 to 0.20, worst for combination bundles and x3.
- **Norepinephrine rms:** 0.03 to 0.11.
- **Measurement-only bundles:** both readouts stay near zero (rms 0.04 and 0.02).

So the readouts are accurate enough to carry a dose surprise of roughly 0.1 units. That is small next to the true per-unit insulin effect (about −2 mg/dL per unit-hour at lag 1, about −10 cumulative). They can feed the planned v2 injection, which adds a dose-surprise term in the core so the rollout responds to edited doses.

The conversation has just restarted and I lost the working context, including how the replica functions are defined. Before building v2 I will re-read the architecture file and the start of my notebook, and list what the kernel still holds. This run changes nothing in the model and uses no forward pass. I am checking two things: whether the core-step replica (step_full and the others) and the readout or gate parameters are still in memory. If so, I can go straight to computing the v2 prelude-FFN edit. If not, I must rebuild those pieces first.

```python
import os, json, types, numpy as np
arch = open('/model/ARCHITECTURE.md').read()
print("ARCH chars", len(arch)); print(arch[:9000])
notes = open('/work/notes.md').read()
print("\n==== NOTES chars", len(notes)); print(notes[:max(0, len(notes)-8000)][:6000])
print("\n==== KERNEL OBJECTS")
fn, arr, oth = [], [], []
for k, v in list(globals().items()):
    if k.startswith('_') or isinstance(v, types.ModuleType): continue
    if callable(v) and hasattr(v, '__code__'): fn.append(k)
    elif isinstance(v, np.ndarray): arr.append(f"{k}{v.shape}")
    elif hasattr(v, 'shape'): arr.append(f"{k}{tuple(v.shape)}*")
    elif isinstance(v, (int, float, str, bool)): oth.append(f"{k}={str(v)[:30]}")
    else: oth.append(f"{k}:{type(v).__name__}")
print("FUNCS:", ", ".join(sorted(fn)))
print("ARRAYS:", ", ".join(sorted(arr))[:3500])
print("OTHER:", ", ".join(sorted(oth))[:2500])
print("edited:", edited())
```