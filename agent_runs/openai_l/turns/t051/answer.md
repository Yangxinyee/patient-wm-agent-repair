The reconstruction error is now essentially zero, confirming the action encoder on the tested bundles. I’ll recover its implementation and the state-update equations before editing. Matching bundles containing infusion ends would confirm that this pathway is safe to use for a dose-sensitive correction.

```python
from pathlib import Path
import inspect
import numpy as np
import pandas as pd

g46 = dict(globals())

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RETAINED FUNCTIONS ===")
for name, obj in g46.items():
    if inspect.isfunction(obj) and not name.startswith("_"):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)

print("\n=== RETAINED DATAFRAMES ===")
for name, obj in g46.items():
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns))

print("\n=== RETAINED VARIABLE NAMES ===")
print(", ".join(
    name for name, obj in g46.items()
    if not name.startswith("_") and not inspect.ismodule(obj)
))

print("\n=== ACTION / LATENT / STATE TENSORS ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ("action", "z_post", "z_prior", "prelude", "gru", "slot", "state")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 46 / restart recovery\n"
        "Run 45 reproduced no-END action posterior means with RMS 2.1e-5. "
        "Critical conventions: float32 bin boundaries, kq 0, empty measurement "
        "encoding, insulin units (3,2), NE units (4,3). Recovering validated "
        "functions and full update equations before a residual-dose edit. "
        "No tensors edited; forward passes 4/50.\n"
    )
```