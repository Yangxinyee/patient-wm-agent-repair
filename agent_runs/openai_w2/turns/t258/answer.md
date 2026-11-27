The last result confirms that the stronger edit was installed, not that it is safe or more accurate. I’ll recover the validation cohorts and model details before testing it. A factual NLL increase above 0.005 would rule out this trial even if its dose responses improve.

```python
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRECORD FORMAT")
print(Path("/data/README.md").read_text())

print("\nCURRENT EDIT")
print(edited())
print("Matches installed trial:",
      np.array_equal(np.asarray(load("core.b_in.weight")),
                     np.asarray(candidate_B219)))
print("Fallback available:", "fallback_B219" in globals())

for k in (26, 27, 28):
    folder = Path(f"/work/fwd_{k}")
    print(f"\nFORWARD {k} FILES:", [p.name for p in folder.iterdir()])
    factual_path = folder / "factual_original.csv"
    if factual_path.exists():
        frame = pd.read_csv(factual_path)
        print("Columns:", frame.columns.tolist())
        stay_col = next(c for c in frame.columns if "stay" in c.lower())
        ids = [int(x) for x in frame[stay_col].drop_duplicates()]
        print("Stay count:", len(ids))
        if k == 26:
            resume_trial_ids = ids
            print("DIFFICULT-COHORT FACTUAL REQUEST:")
            print(json.dumps({"type": "factual", "model": "both",
                              "split": "tuning", "stays": ids}))
        print(frame.head(2).to_string(index=False))

print("\nRELEVANT CACHED OBJECTS")
print([n for n in globals()
       if any(s in n.lower() for s in
              ("target", "effect", "rho", "causal", "iv_", "summary217",
               "summary218", "candidate_b", "delta_ins", "delta_ne"))][-65:])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run220: stronger-trial validation recovery\n"
            "No weight changes. Recovering architecture, exact difficult-cohort "
            "IDs, and cached evidence before paired factual testing. "
            "candidate_B219 remains provisional; fallback_B219 is preserved.\n")
```
