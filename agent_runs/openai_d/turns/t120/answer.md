The edit substantially improves both insulin responses, but they remain smaller than the provisional targets. Norepinephrine responses are essentially unchanged. I’ll preserve this candidate and recover the fitting helpers and causal evidence before strengthening it. A stronger correction is worthwhile only if it remains record-supported and within the factual NLL budget.

```python
from pathlib import Path
import numpy as np
import torch

r107_backup = {
    n: torch.as_tensor(load(n)).detach().cpu().clone()
    for n in edited()
}
np.savez("/work/insulin_candidate_v1.npz",
         **{n: t.numpy() for n, t in r107_backup.items()})

with open("/work/notes.md", "a") as f:
    f.write("""
## Forward 10–11: first validated insulin candidate
Four value-head tensors, hash 70c11c43ec. Paired tuning factual NLL:
original 1.63461, edited 1.63592, delta +0.00131 (4751 readings).
Dose shifts on the same 100 stays:
original insulin -1.847/+0.777; edited -5.908/+2.782.
Original NE -0.498/-0.600; edited -0.517/-0.615.
Insulin intervention transmission improved substantially, but remains below
provisional record-derived targets -13.599/+6.800. NE is not repaired.
Backed up this factual-budget-compliant candidate in memory and
/work/insulin_candidate_v1.npz before any further changes.
""")

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())
print("=== EARLY NOTEBOOK: CAUSAL EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:12000])
print("=== PERSISTENT HELPERS ===")
print(sorted(n for n, v in list(globals().items())
             if callable(v) and not n.startswith("_")))
print("=== RECENT VARIABLES ===")
for n, v in sorted(list(globals().items())):
    if any(n.startswith(p) for p in
           ["r99", "r100", "r101", "r102", "r103", "r104", "r105", "r106"]):
        if isinstance(v, dict):
            print(n, "dict", list(v.keys())[:35])
        elif hasattr(v, "shape"):
            print(n, type(v).__name__, tuple(v.shape))
        elif isinstance(v, (str, int, float, list, tuple)):
            print(n, type(v).__name__, str(v)[:450])
        elif callable(v):
            print(n, "callable")
        else:
            print(n, type(v).__name__)
```