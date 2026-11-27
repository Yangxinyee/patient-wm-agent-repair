The needed prelude and recurrent replay helpers are still available. The last inspection stopped on a formatting error, before changing any weights. I’ll now check their return values and verify the drug re-encoder against the recovered factual inputs. Near-zero encoding error and reproducible dose contrasts would justify fitting a correction; a mismatch would require fixing the replay first.

```python
import numpy as np
import torch

print("EDITS", edited())
print("Prelude locals:", prelude81.__code__.co_varnames)
print("Action encoder locals:", fn76.__code__.co_varnames)
print("State update locals:", state_update82.__code__.co_varnames)

p86 = prelude81(h81[:3], e81[:3], u81[:3],
                posterior_mean(ctx81[:3], e81[:3], u81[:3]))
print("PRELUDE RETURN", type(p86).__name__)
if isinstance(p86, (tuple, list)):
    for j, x in enumerate(p86):
        print(j, type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else str(x)[:100])
else:
    print(tuple(p86.shape))

a86 = fn76("clear_all", (1.0, 1.0))
print("ACTION ENCODER RETURN", type(a86).__name__)
if isinstance(a86, (tuple, list)):
    for j, x in enumerate(a86):
        print(j, type(x).__name__, tuple(x.shape) if hasattr(x, "shape") else str(x)[:100])
else:
    print(a86.shape)
    aa86 = torch.as_tensor(a86[idx81], dtype=torch.float32)
    print("ACTION RMSE", float(torch.mean((aa86-u81)**2).sqrt()))
    print("DRUG ACTION RMSE", float(torch.mean((aa86[active81]-u81[active81])**2).sqrt()))

print("RATE LABELS", rates81.shape,
      np.quantile(rates81, [0, .5, .9, .99, 1], axis=0))
print("DIRECTION/FIT ARRAYS")
for k in sorted(list(globals())):
    v = globals()[k]
    if any(s in k.lower() for s in ("dir", "coef", "fit", "slope")) and isinstance(v, (np.ndarray, torch.Tensor)):
        print(k, tuple(v.shape))

with open("/work/notes.md", "a") as f:
    f.write("\nPython runs 85-86; forwards 7. Run 85 stopped on a diagnostic formatting TypeError, without any weight changes. Verified retained replay interfaces and checked factual action re-encoding before fitting a dose-contrast repair. Original model remains installed.\n")
```