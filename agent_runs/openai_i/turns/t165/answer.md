The architecture confirms that actions enter the shared recurrent state, but the inventory failed before recovering the causal estimates. I’ll fix that inspection. The estimates must establish the targets before we choose any direction or gain for an edit.

```python
from pathlib import Path
import numpy as np
import torch
import pandas as pd

print("=== RETAINED ARRAYS, TABLES, DICTIONARIES, AND HELPERS ===")
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"In", "Out", "NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), list(v.keys())[:18])
    elif callable(v) and getattr(v, "__module__", "") == "__main__":
        print(k, "function")

notes148 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(notes148[:17000])
print("\n=== LATER IDENTIFICATION AND NUMERICAL EVIDENCE ===")
lines148 = [
    line for line in notes148.splitlines()
    if any(term in line.lower() for term in
           ["target", "instrument", "causal", "impulse", "random",
            "identif", "nll:", "x2", "slope", "saved"])
]
print("\n".join(lines148)[-14000:])

print("\n=== AVAILABLE FORWARD ARTIFACTS ===")
for p in sorted(Path("/work").glob("fwd_*")):
    print(p.name, [(q.name, q.stat().st_size) for q in p.iterdir()])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 148 — bounded recovery\n"
        "- Architecture reread in Python 147; its inventory failed on a "
        "module exposing a callable shape attribute.\n"
        "- Corrected inventory types; recovering causal evidence and "
        "available forward artifacts. No edits active.\n"
    )
```
