The rollback succeeded. The recovered notes support negative insulin and positive norepinephrine effects, but reject the early state-space estimates. I’ll recover the later effect estimates, candidate scores, and unused audit IDs. These should clarify the appropriate response scale and whether the restored candidate has enough factual-fit margin.

```python
import pandas as pd
import json
import re
import numpy as np
import torch
from pathlib import Path

text293 = Path("/work/notes.md").read_text()
sections293 = text293.split("\n## ")
print("=== LATER IDENTIFICATION NOTES ===")
chosen293 = []
for s in sections293:
    title = s.splitlines()[0] if s else ""
    if re.search(r"Run(?:s)?\s+(17|18|19|20|21|22|23|24|25|26|27|28|29|30)\b", title):
        chosen293.append(s)
print("\n## ".join(chosen293)[:4500])

print("=== CURRENT CANDIDATE STRUCTURE ===")
v293 = grid286[(.35, .3, 8)]
if isinstance(v293, dict):
    for k, v in v293.items():
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
            if np.prod(v.shape) <= 16:
                print(v)
        else:
            print(k, str(v)[:450])
else:
    print(type(v293).__name__)
    for j, v in enumerate(v293):
        print(j, type(v).__name__, getattr(v, "shape", None),
              str(v)[:350] if not hasattr(v, "shape") or np.prod(v.shape) <= 16 else "")

print("=== GRID METRICS ===")
for k, v in grid286.items():
    if isinstance(v, dict):
        small = {}
        for name, x in v.items():
            if hasattr(x, "shape"):
                if np.prod(x.shape) <= 16:
                    small[name] = torch.as_tensor(x).tolist()
            elif isinstance(x, (float, int, tuple, list)):
                small[name] = x
        print(k, str(small)[:350])

print("=== RELEVANT HELPER NAMES ===")
print([k for k in sorted(list(globals()))
       if re.search(r"weighted_fit|nll228|slope|target|effect|quiet", k, re.I)])

tuning_ids293 = pd.read_csv("/data/tuning.csv", usecols=["stay_id"])["stay_id"].drop_duplicates().tolist()
assert len(tuning_ids293) == 400
audit0_293 = tuning_ids293[:100]
audit2_293 = tuning_ids293[200:300]
print("=== FACTUAL AUDIT COHORT 0 ===")
print(json.dumps(audit0_293))
print("=== FACTUAL AUDIT COHORT 2 ===")
print(json.dumps(audit2_293))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 293 / bounded final-audit preparation\n"
            "No weight changes. Recovered later causal notes, grid286 candidate "
            "metadata, and factual audit cohorts 0–99 and 200–299 from tuning.csv.\n")
```