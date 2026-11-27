The architecture confirms that the edit removes dose magnitude information while retaining drug identity, presence, and timing. I’ll now aggregate the paired factual tests and identify the untested tuning stays. A small aggregate NLL change across all 400 stays would support factual preservation; a concentrated deterioration would require revising the edit.

```python
import pandas as pd
import numpy as np
from pathlib import Path

print("=== KERNEL VARIABLE NAMES ===")
print(sorted(k for k in globals() if not k.startswith("_")))

print("=== NOTEBOOK METHODS AND CONCLUSIONS ===")
for line in Path("/work/notes.md").read_text().splitlines():
    if line.strip() and not line.lstrip().startswith(("[", "{")):
        print(line[:450])

factual_pairs = []
covered_tuning_ids = set()
print("=== FACTUAL FILE AUDIT ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    op = folder / "factual_original.csv"
    ep = folder / "factual_edited.csv"
    if not (op.exists() and ep.exists()):
        continue
    fo = pd.read_csv(op)
    fe = pd.read_csv(ep)
    print(folder.name, "columns:", list(fo.columns), "rows:", len(fo), len(fe))
    assert len(fo) == len(fe)
    keys = [c for c in ("stay_id", "hour", "event") if c in fo.columns]
    assert keys and fo[keys].equals(fe[keys]), "Paired rows are not aligned"
    nll_cols = [c for c in fo.columns if "nll" in c.lower()]
    for c in nll_cols:
        delta = fe[c].to_numpy() - fo[c].to_numpy()
        print(c, "original:", fo[c].mean(),
              "edited:", fe[c].mean(), "delta:", delta.mean())
    if "stay_id" in fo:
        ids_here = set(fo.stay_id.unique())
        print("stays:", len(ids_here), "id range:", min(ids_here), max(ids_here))
        covered_tuning_ids.update(ids_here)
    factual_pairs.append((folder.name, fo, fe, nll_cols))

tuning_ids_all = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"]).stay_id.unique().tolist())
remaining_tuning_ids = [s for s in tuning_ids_all if s not in covered_tuning_ids]
print("=== COVERAGE ===", len(covered_tuning_ids), "of", len(tuning_ids_all))
print("=== REMAINING TUNING IDS ===")
print(remaining_tuning_ids)
```