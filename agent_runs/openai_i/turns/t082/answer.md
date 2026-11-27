The atom encoders agree to numerical precision, but the posterior mismatch shows that bundle assembly is still wrong. I’ll inspect the architecture and retained helpers before making an edit. Matching native posterior states would validate the reconstruction; continued disagreement would rule it out as a basis for calibration.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            signature = str(inspect.signature(obj))
        except Exception:
            signature = "(signature unavailable)"
        print(name, signature)

print("\n=== RETAINED DATA / STATE OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_"):
        continue
    if isinstance(obj, pd.DataFrame) and len(obj) > 100:
        print(name, obj.shape, list(obj.columns))
    elif isinstance(obj, np.lib.npyio.NpzFile):
        print(name, {k: obj[k].shape for k in obj.files})
    elif isinstance(obj, dict) and any(
        term in name.lower()
        for term in ("state", "helper", "effect", "causal", "impulse", "recon", "post")
    ):
        print(name, "keys:", list(obj.keys())[:30])

print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 74; forward 6 — reconstruction safeguard\n"
        "- Atom and empty encoders agree numerically, but full posterior inputs "
        "remain inconsistent, especially on observation-only bundles.\n"
        "- No weights edited. Recovering architecture and retained helpers before "
        "calibrating a shared-state dose correction.\n"
    )
```