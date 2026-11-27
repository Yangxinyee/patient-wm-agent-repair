The latest pass ran a diagnostic probe, not a candidate repair; its NLL is unacceptable for submission. I’ll restore the original weights and use the saved outputs to identify the head’s actual input layout. Exact numerical agreement would validate the reconstruction; another mismatch would rule it out.

```python
from pathlib import Path
import json, numpy as np, pandas as pd, torch

r87_probe_weights = {n: load(n).clone() for n in edited()}
for n in list(edited()):
    drop_edit(n)
print("Remaining edits:", edited())

print("\n=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

r87_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(r87_notes[:11000])

r87_probe = pd.read_csv("/work/fwd_9/factual_edited.csv")
print("\n=== PROBE COLUMNS / SAMPLE ===")
print(r87_probe.columns.tolist())
print(r87_probe.head(12).to_string(index=False))

print("\n=== PERSISTENT HELPERS AND ARRAYS ===")
for n in sorted(list(globals())):
    if n.startswith("_") or n.startswith("r87"):
        continue
    v = globals()[n]
    if callable(v):
        print(n, "callable")
    elif isinstance(v, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(n, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(n, "dict", len(v), "keys:", list(v)[:12])
    elif isinstance(v, (list, tuple)) and len(v) < 25:
        print(n, type(v).__name__, str(v)[:220])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 87: restart recovery\n"
            "Forward 9 was an artificial register probe (NLL 2.62934), not a candidate repair. "
            "Saved its diagnostic weights in r87_probe_weights and reverted every edit. "
            "Recovering the exact head layout from its saved outputs before further edits.\n")
```