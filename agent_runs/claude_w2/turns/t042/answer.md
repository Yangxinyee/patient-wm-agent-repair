Run 37 re-read the architecture. The kernel still holds:
- my record-based estimates: Kins, Knor, the deconvolution helpers;
- the per-reading factual output of the original model on 100 tuning stays (fwd_2);
- states from fwd_3;
- a partial re-implementation of the model's front end (atom embedding, set encoders, the z-posterior input order, confirmed in run 36).

A key point from the architecture: dose information reaches glucose and MAP predictions only through the state. The output head's "last value" register is per code, so a glucose prediction never reads the drug directly. Doses enter as action atoms (numeric z_dose and z_rate plus dose and rate bins), then u_k, then the z posterior and the prelude.

That shapes the fix. The model's dose response is about a fifth of what the records imply. Since it fits the record well, it has probably folded the missing insulin effect into how it extrapolates from the glucose level itself. If I simply amplify the dose pathway, the effect would be counted twice in factual predictions, and factual NLL would likely get worse.

The repair that should keep the factual fit and get dose edits right adds the missing effect only for the dose's departure from what the policy would give at that glucose level. Before designing where to put this in the weights, I need the exact tensor layout: encoders, z posterior, prelude, recurrent cell, c_out and coda, and the value head. Next I list all tensor names and shapes plus the vocab config.

```python
import json
by = {}
for nm_ in NAMES:
    top = nm_.split('.')[0] + '.' + nm_.split('.')[1] if nm_.count('.')>=1 else nm_
    by.setdefault(top, []).append(f"{nm_[len(top)+1:] or '.'}{list(SHAPES[nm_])}")
for kk, vv in by.items():
    print(kk, ': ', '; '.join(vv))
print(VOCAB['config'])
print(list(VOCAB.keys()))
```