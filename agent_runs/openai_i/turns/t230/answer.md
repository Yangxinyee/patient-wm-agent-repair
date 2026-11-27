The neutral candidate worsened factual NLL by 0.01741, exceeding the allowed 0.005 before adding any drug effect. I’ll revert it and inspect the retained pathway calculations. The next design must disturb factual processing less; accurate dose transport alone is insufficient.

```python
from pathlib import Path
import numpy as np

for name in list(edited()):
    drop_edit(name)

with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 17; Python 205 — reject neutral204\n"
            "- Neutral private-channel candidate changed tuning factual NLL from 1.63155 to 1.64896 (+.01741), failing the .005 safeguard before drug-response weights were added.\n"
            "- Restored all original tensors. Accurate innovation transport did not imply factual preservation.\n"
            "- Inspecting retained calculations and architecture before attempting a less disruptive transport edit.\n")

print("ACTIVE EDITS:", edited())
print("\nARCHITECTURE:\n", Path("/model/ARCHITECTURE.md").read_text())
notes205 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK:\n", notes205[:18000])
print("\nRETAINED OBJECTS:")
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"In", "Out", "get_ipython", "exit", "quit"}:
        continue
    if isinstance(value, np.ndarray):
        print(key, "ndarray", value.shape, value.dtype)
    elif hasattr(value, "shape") and type(value).__module__.startswith("torch"):
        print(key, "tensor", tuple(value.shape))
    elif callable(value) and getattr(value, "__module__", "") == "__main__":
        import inspect
        try:
            print(key, "function", inspect.signature(value))
        except Exception:
            pass
    elif isinstance(value, dict):
        print(key, "dict", len(value), "keys:", list(value)[:12])
    elif isinstance(value, (list, tuple)) and len(value) < 30:
        print(key, type(value).__name__, str(value)[:350])
    elif isinstance(value, (int, float, str)) and not key.startswith("notes"):
        print(key, str(value)[:180])
```