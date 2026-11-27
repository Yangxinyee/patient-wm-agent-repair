The paired-state readouts predict held-out edit responses reasonably well, but the current drug-token directions still miss the record-based effect estimates. I’ll recover the exact interfaces before replacing those directions. A replacement should produce the estimated responses across both measurement types while preserving factual predictions; otherwise I’ll reject it.

```python
from pathlib import Path
import inspect, dis
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RELEVANT FUNCTION INTERFACES ===")
for name in [
    "r59_core", "repaired_core_local", "fn93",
    "audit_block", "norm0", "mlp0",
]:
    obj = globals().get(name)
    if callable(obj):
        print(name, inspect.signature(obj))

print("\n=== EXACT LOCAL UPDATE HELPERS ===")
for name in ["r59_core", "repaired_core_local", "fn93"]:
    obj = globals().get(name)
    if callable(obj):
        print("\nFUNCTION:", name)
        dis.dis(obj)

print("\n=== VALIDATED READOUT MODELS ===")
for name in ["models145", "diagnostics145", "models144", "diagnostics144"]:
    obj = globals().get(name)
    print("\n", name, type(obj).__name__)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict):
                print(key, {
                    k: (v if np.isscalar(v) or isinstance(v, str)
                        else (type(v).__name__, getattr(v, "shape", None)))
                    for k, v in value.items()
                })
            else:
                print(key, repr(value)[:1800])
    else:
        print(repr(obj)[:4000])

print("\n=== RECENT READOUT/ROUTING VARIABLES ===")
for name, obj in list(globals().items()):
    if any(tag in name for tag in ["143", "144", "145", "146", "59", "CF89"]):
        if callable(obj):
            try:
                print(name, "callable", inspect.signature(obj))
            except Exception:
                pass
        elif isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:25])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length:", len(obj))
        elif isinstance(obj, (str, int, float, bool)):
            print(name, repr(obj)[:250])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 147: recover validated routing interfaces\n"
            "- Current candidate has factual margin but underestimates both drug effects.\n"
            "- Recovering exact local update, latent candidate, and paired readout interfaces "
            "to replace unvalidated token directions; no edits changed.\n"
            "- Python runs: 147; forward passes: 10.\n")
```