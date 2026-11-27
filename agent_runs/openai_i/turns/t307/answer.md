The corrected route produces much larger dose responses, but their magnitude and doubling-versus-halving asymmetry still need checking against the records. I’ll recover the record-based effect estimates and separate the routing changes from the response weights. The goal is to retain useful responses while gaining factual-fit margin.

```python
import re, inspect, numpy as np
from pathlib import Path

print("=== RECORD-BASED IDENTIFICATION NOTES ===")
text270 = Path("/work/notes.md").read_text()
sections270 = re.split(r"(?m)(?=^## )", text270)
terms270 = ("identif", "persistence", "g-computation", "causal target",
            "effect estimate", "record estimates", "ground truth", "estimated shift")
selected270 = [s for s in sections270
               if any(t in s.lower() for t in terms270)]
budget270 = 9500
for s in selected270:
    if budget270 <= 0:
        break
    part = s[:min(1600, budget270)]
    print(part)
    budget270 -= len(part)

print("=== RESPONSE CANDIDATE CONSTRUCTION ===")
fun270 = response_candidate231
print("names", fun270.__code__.co_names)
print("constants", fun270.__code__.co_consts)
for k in fun270.__code__.co_names:
    if k in globals() and k not in ("np", "torch"):
        v = globals()[k]
        if isinstance(v, dict):
            print(k, "dict", [(a, getattr(b, "shape", None)) for a, b in v.items()][:14])
        elif isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, "shape", tuple(v.shape),
                  "norm", float(np.linalg.norm(np.asarray(v))))
        elif isinstance(v, (str, int, float, list, tuple)):
            print(k, v)

print("=== NEUTRAL ROUTE VERSUS RESPONSE ===")
base270 = {k: v.copy() for k, v in np.load("/work/small_route222.npz").items()}
for name in sorted(corrected269):
    orig = np.asarray(load_original(name))
    a = corrected269[name]
    b = base270.get(name, orig)
    print(name, "shape", a.shape,
          "base-original norm", round(float(np.linalg.norm(b-orig)), 6),
          "response-base norm", round(float(np.linalg.norm(a-b)), 6))
print("=== SAVED RESPONSE ARRAY KEYS ===")
for file in ("response_directions231.npz", "response_candidates216.npz",
             "response_fd230.npz"):
    with np.load("/work/" + file) as a:
        print(file, [(k, a[k].shape) for k in a.files])
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 25–26; Python 270 — corrected route nearly fits and responds\n"
            "- corrected269 factual dNLL +.00546 on first 100 tuning stays: narrowly fails .005.\n"
            "- Dose shifts: insulin x2 -10.857, half +3.275; NE x2 +1.755, half -2.066.\n"
            "- Original on the same stays: -2.016, +1.051, +.309, -.070.\n"
            "- Recovering record-based targets and isolating neutral routing footprint before refinement.\n"
            "- corrected269 remains provisional, not accepted.\n")
```