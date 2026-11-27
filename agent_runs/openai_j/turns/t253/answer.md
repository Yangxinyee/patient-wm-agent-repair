The stronger edit worsened factual NLL by **0.00432 nats**, narrowly within the 0.005 limit on these stays. Its rollout benefit remains untested. I’ll recover the checkpoints and evaluation IDs; better dose responses with a reliable factual margin would justify keeping it.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

audit226_notes = Path("/work/notes.md").read_text()
print("\nEARLIER EVIDENCE AND CONSTRUCTION")
print(audit226_notes[:18000])

audit226_factual = pd.read_csv("/work/fwd_22/factual_original.csv")
print("\nFACTUAL COLUMNS:", audit226_factual.columns.tolist())
print(audit226_factual.head(3).to_string(index=False))
audit226_idcol = next(
    c for c in ("stay_id", "stay", "icustay_id") if c in audit226_factual.columns
)
audit226_stays = audit226_factual[audit226_idcol].drop_duplicates().tolist()
print("\nEXACT FORWARD-22 STAYS:", audit226_stays)

print("\nRETAINED HELPERS AND ARRAYS")
for audit226_name in sorted(list(globals())):
    if audit226_name.startswith(("head177", "map192", "map224", "map225",
                                 "cal190", "insulin", "kernel", "target")):
        audit226_obj = globals()[audit226_name]
        print(audit226_name, type(audit226_obj).__name__,
              getattr(audit226_obj, "shape", ""),
              list(audit226_obj.keys())[:15] if isinstance(audit226_obj, dict) else "")

print("\nCHECKPOINT CONTENTS")
for audit226_file in ("validated_forward21.npz", "map_gain8_candidate225.npz",
                      "best_insulin_v1.npz", "best_combined_v1.npz"):
    with np.load("/work/" + audit226_file) as audit226_ckpt:
        print(audit226_file,
              [(k, audit226_ckpt[k].shape) for k in audit226_ckpt.files])
print("\nCURRENT EDITS:", edited())
```
