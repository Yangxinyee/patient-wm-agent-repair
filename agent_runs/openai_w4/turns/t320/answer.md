The shared-gate candidate nearly erased the norepinephrine response, so it failed the rollout hypothesis. That is a model-transmission failure, not evidence that the ICU’s drug effect is absent. I’ll restore the prior edit and inspect the gate; opening it selectively must restore both dose responses without harming factual fit.

```python
import dis
import inspect
import numpy as np
import torch
from pathlib import Path

for n, v in backup275.items():
    save_edit(n, v.clone())

print("RESTORED PRE-275 SUBMISSION")
for label, candidate in [
    ("restored", backup275),
    ("failed_symmetric", candidate275)
]:
    w = a262(candidate["z_post.0.weight"])
    b = a262(candidate["z_post.0.bias"])
    o = a262(candidate["z_post.2.weight"])
    print(label)
    for r in (156, 89):
        print("row", r, "bias", float(b[r]),
              "ctx/e/action norms",
              [float(np.linalg.norm(w[r, j:j+192])) for j in (0, 192, 384)],
              "output norm", float(np.linalg.norm(o[:, r])))

print("\nFACTUAL EVALUATOR")
print(inspect.signature(factual_eval194))
print("dependencies", factual_eval194.__code__.co_names)
print("simple constants",
      [v for v in factual_eval194.__code__.co_consts
       if isinstance(v, (str, int, float, tuple))][:35])

print("\nINSULIN CONSTRUCTOR")
print("dependencies", symmetric_ins265.__code__.co_names)
dis.dis(symmetric_ins265)

print("\nPOTENTIAL POSTERIOR INPUT CACHES")
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    shape = getattr(v, "shape", ())
    if (len(shape) == 2 and shape[1] in (576, 992)
        or any(s in k.lower() for s in ("ctx", "post_input", "gate", "packed"))):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), list(v)[:8])

print("\nRECENT NE CONSTRUCTION NOTES")
lines276 = Path("/work/notes.md").read_text().splitlines()
matches276 = [s for s in lines276 if
              any(t in s for t in ("156", "89", "NE", "norepinephrine"))]
for s in matches276[-12:]:
    print(s[:900])

with open("/work/notes.md", "a") as f:
    f.write("\n## GPU pass 31; Run 276\n"
            "Shared-gate NE gain 1/exact opposing columns failed rollout calibration: "
            "reached-row shifts insulin x2 -11.655, half +9.265; "
            "NE x2 +.072, half -.513. Restored backup275 (pre-275 submission). "
            "This rejects that gate setting, not the record-based NE effect. "
            "Inspecting selective gate opening before further CPU screening.\n")
```