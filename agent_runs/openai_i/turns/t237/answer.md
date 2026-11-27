The neutral edit preserved factual predictions on the tested stays almost exactly. That supports the routing construction, but does not yet establish a drug effect. Next I’ll recover its implementation and the record-derived estimates before calibrating the two response channels. A useful channel must change dose responses while keeping factual loss within the limit.

```python
from pathlib import Path
import numpy as np
import torch
import inspect
import re

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes211 = Path("/work/notes.md").read_text()
sections211 = re.split(r"(?m)(?=^## )", notes211)
print("\nEARLIER RECORD-EVIDENCE NOTES")
matches211 = [
    s for s in sections211
    if re.search(r"confound|identif|instrument|policy|record-derived|slope|innovation", s, re.I)
]
for s211 in matches211[:8] + matches211[-8:]:
    print(s211[:1800])

print("\nRECENT RETAINED OBJECTS")
for name211 in sorted(list(globals())):
    match211 = re.search(r"(\d+)$", name211)
    if match211 and 194 <= int(match211.group(1)) <= 209:
        obj211 = globals()[name211]
        if callable(obj211):
            print(name211, "FUNCTION",
                  getattr(getattr(obj211, "__code__", None), "co_varnames", None),
                  getattr(getattr(obj211, "__code__", None), "co_names", None))
        elif hasattr(obj211, "shape"):
            print(name211, type(obj211).__name__, tuple(obj211.shape))
        elif isinstance(obj211, dict):
            print(name211, "dict", len(obj211), "keys", list(obj211)[:8])
        elif isinstance(obj211, (list, tuple)):
            print(name211, type(obj211).__name__, len(obj211))
        elif isinstance(obj211, (str, int, float)):
            print(name211, repr(obj211)[:160])

print("\nCORE HELPER CONTRACTS")
for name211 in ["mlp136", "call_core161", "batch_q182", "batch_replay182"]:
    fn211 = globals().get(name211)
    if callable(fn211):
        print(name211, inspect.signature(fn211))
        print("globals:", fn211.__code__.co_names)
        try:
            print(inspect.getsource(fn211))
        except (OSError, TypeError):
            pass

print("\nCURRENT EDIT GEOMETRY")
for name211 in edited():
    a211 = load(name211)
    b211 = load_original(name211)
    delta211 = np.asarray(a211) - np.asarray(b211)
    print(name211, tuple(a211.shape),
          "delta RMS", float(np.sqrt(np.mean(delta211**2))),
          "delta max", float(np.max(np.abs(delta211))))
print("z_in innovation-column norms:",
      np.linalg.norm(np.asarray(load("z_in.weight"))[:, [16, 12]], axis=0))

with open("/work/notes.md", "a") as f211:
    f211.write(
        "\n## Python 211 — recover neutral-channel calibration context\n"
        "- Neutral209 passed the first 100-stay factual check at near-identical precision.\n"
        "- Recovering architecture, record evidence and private-channel helper contracts.\n"
        "- No response weights added; causal rollout performance remains unvalidated.\n"
    )
```