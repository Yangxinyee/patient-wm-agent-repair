The last diagnostic showed that END atoms still carry dose signals, and combined WORLD+END bundles can amplify them unevenly. That could contribute to the rollout asymmetry, but it is not yet proof. I’ll check the model’s bundle semantics and the current transport code; the hypothesis holds only if actual record bundles distort exposure in the same way.

```python
from pathlib import Path
import inspect, re, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

print("\n=== RECOVERABLE FUNCTIONS ===")
_g251 = dict(globals())
for _k, _v in sorted(_g251.items()):
    if inspect.isfunction(_v) and getattr(_v, "__module__", "") == "__main__":
        print(_k, inspect.signature(_v), "uses:", _v.__code__.co_names[:35])

print("\n=== RELEVANT OBJECTS ===")
for _k, _v in sorted(_g251.items()):
    if re.search(r"(action|transport|center|candidate|causal|panel|coef|gate|pair|exposure|cache|w184|w193|state)", _k, re.I):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_k, type(_v).__name__, tuple(_v.shape))
        elif isinstance(_v, dict):
            print(_k, "dict", list(_v)[:25])
        elif not inspect.isfunction(_v):
            print(_k, type(_v).__name__, str(_v)[:350])

_notes251 = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION AND DESIGN NOTES ===")
print(_notes251[:18000])
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Run 251: bundle-semantics audit\n"
             "- END and WORLD+END synthetic signals are irregular; these are not yet validated explanations of rollout asymmetry.\n"
             "- No weights changed. Re-reading architecture and recovering exact action/transport helpers before another trial.\n")
```