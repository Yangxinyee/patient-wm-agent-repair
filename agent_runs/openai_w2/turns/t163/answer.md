The stronger edit exceeded the factual-fit limit, so the smaller validated edit remains installed. I’ll check which drug caused the extra NLL and revisit the record-based effect estimates. If most of the cost came from insulin, strengthening norepinephrine separately may improve the readouts without breaking the fit limit; otherwise, we’ll need a more selective edit.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== EARLY RESEARCH NOTES ===")
notes142 = Path("/work/notes.md").read_text()
print(notes142[:18000])
print("=== CURRENT EDIT ===", edited())
for k142 in ["Wdose134", "V128", "fallback_recurrent139", "fit128",
             "repair_head_cfg", "raw_coef134", "affine_b134"]:
    if k142 in globals():
        v142 = globals()[k142]
        if isinstance(v142, dict):
            print(k142, {k: (getattr(v, "shape", None) if hasattr(v, "shape") else str(v)[:2000])
                         for k, v in v142.items()})
        else:
            print(k142, type(v142).__name__, getattr(v142, "shape", None))
print("=== FACTUAL FILE SCHEMAS ===")
for f142 in ["/work/fwd_13/factual_original.csv",
             "/work/fwd_13/factual_edited.csv",
             "/work/fwd_15/factual_edited.csv"]:
    if Path(f142).exists():
        df142 = pd.read_csv(f142)
        print(f142, df142.shape, df142.columns.tolist())
        print(df142.head(3).to_string(index=False))
print("=== EFFECT / COHORT OBJECTS ===")
print([k for k in globals() if any(s in k.lower() for s in
      ["target", "causal", "effect", "latent_record", "tuning", "stay_ids", "iv_result"])])
```
