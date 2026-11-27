The probe confirms that drug-start encodings are correct, but empty-action and observation bundles still need reconstruction. No edits remain installed. I’ll recover the update equations and retained helpers before targeting the effective action pathway. Agreement with the probe would validate that reconstruction; disagreement would rule out using it to design an edit.

```python
from pathlib import Path
import numpy as np
import torch

arch_text = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch_text)
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

print("=== RETAINED FUNCTIONS ===")
print([
    k for k, v in list(globals().items())
    if callable(v) and getattr(v, "__module__", None) == "__main__"
])

print("=== RELEVANT RETAINED OBJECTS ===")
for k, v in list(globals().items()):
    if any(s in k.lower() for s in
           ("prelude", "context", "repair", "probe", "kernel", "causal",
            "policy", "state", "bundle", "r67", "r74")):
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys:", list(v)[:12])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
        elif callable(v):
            print(k, "function")
        elif isinstance(v, (str, int, float)):
            print(k, repr(v)[:180])

print("=== UPDATE/READOUT TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("prelude", "core", "readout", "head", "decode", "out.", "action")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 75; forwards 7. Original weights remain restored. "
        "Probe establishes exact drug-start encodings and reveals nonzero "
        "empty-action encoding. Recovering update equations and retained "
        "helpers to reconstruct the effective action/recurrent pathway; "
        "posterior-only correction was ineffective because its attention was negligible.\n"
    )
```