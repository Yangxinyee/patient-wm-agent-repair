The probes recover prescribed doses much better than deviations from the dosing policy. That suggests a value-head edit alone could amplify confounding. I’ll inspect the action and recurrent pathways to find where dose information is lost. A distinct, editable action pathway would support a targeted repair.

```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED ANALYSIS OBJECTS ===")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    obj = globals()[key]
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(key, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(key, "dict", len(obj), "keys:", list(obj)[:14])
    elif hasattr(obj, "shape") and hasattr(obj, "columns"):
        print(key, type(obj).__name__, obj.shape, "columns:", list(obj.columns)[:24])
    elif isinstance(obj, (list, tuple)):
        print(key, type(obj).__name__, len(obj))
    elif isinstance(obj, (int, float, str)):
        print(key, repr(obj)[:250])
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(key, "function")

print("\n=== NON-EMBEDDING TENSORS ===")
for name in NAMES:
    if not name.startswith(("embed.", "encode.")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 14: restart recovery and intervention-path hypothesis\n"
        "Downstream probes encode policy-associated dose but almost no residual dose; "
        "a head-only edit is unlikely to recover intervention sensitivity reliably. "
        "Recovering exact recurrent/action architecture and retained statistical objects "
        "before designing a dose-specific, factually compensated edit. No weights edited.\n"
        "Python runs: 14. Forward requests: 3.\n"
    )
```