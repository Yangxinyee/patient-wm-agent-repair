The compact check failed because the validation-stay variable was never created, so the previous run’s completion is uncertain. I’ll verify and reinstall the saved 80% candidate, then read the stay IDs directly from the factual output. This is a bookkeeping repair; only the subsequent GPU test can establish whether the candidate preserves factual fit.

```python
from pathlib import Path
import numpy as np, pandas as pd

b280 = dict(np.load("/work/neutral209.npz"))
c280 = dict(np.load("/work/corrected269.npz"))
cand280 = {k:v.copy() for k,v in c280.items()}
cand280["z_in.weight"] = (
    b280["z_in.weight"] + np.float32(0.8) *
    (c280["z_in.weight"] - b280["z_in.weight"]))
for k in list(edited()):
    drop_edit(k)
for k,v in cand280.items():
    save_edit(k,v)
np.savez("/work/gain278.npz", **cand280)
print("INSTALLED", edited())

f280 = pd.read_csv("/work/fwd_28/factual_original.csv")
print("CSV_COLUMNS", f280.columns.tolist())
idcol280 = next((k for k in f280.columns
    if k.lower() in ["stay_id","stay","id","sid"]
    or "stay" in k.lower()), f280.columns[0])
dev_ids280 = f280[idcol280].drop_duplicates().tolist()
print("VALIDATION_STAYS", dev_ids280)
print("CAUSAL_TARGETS215", repr(globals().get("causal_targets215"))[:1000])
n280 = Path("/work/notes.md").read_text()
p280 = n280.find("## Python 29;")
print("RECORD_TARGET_TABLE", n280[p280:p280+1300] if p280 >= 0 else "not found")
print("RESPONSE_COLUMN_NORMS", [
    (j,round(float(np.linalg.norm(c280["z_in.weight"][:,j]
                                  -b280["z_in.weight"][:,j])),3))
    for j in range(32) if np.linalg.norm(
        c280["z_in.weight"][:,j]-b280["z_in.weight"][:,j]) > 1e-6])
with open("/work/notes.md","a") as f:
    f.write("\n## Python 280 — definitive gain-0.8 installation\n"
            "- Installed corrected269 with z_in response delta scaled by 0.8.\n"
            "- Retrieved factual CSV stay IDs directly; candidate remains provisional.\n")
```
