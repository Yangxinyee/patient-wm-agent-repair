The paired-surrogate route also failed when it required very large token changes: the nonlinear model saturated. I’ll instead test a modest sign reversal of the current small directions, since their official rollout moved both doubled-dose effects away from the record estimates. Improvement under reversal, within the factual budget, would support calibrating these directions with official tests.

```python
import torch, numpy as np
from pathlib import Path

backup199 = {n: load(n).detach().clone() for n in edited()}
base199 = {n: t.detach().clone() for n, t in candidates90[0.0].items()}
candidate199 = {n: t.detach().clone() for n, t in base199.items()}

# Reverse and double only the two added drug-routing directions.
# The compressed posterior and zero-correction baseline remain unchanged.
for j in (30, 31):
    increment199 = candidate193["z_in.weight"][:, j] - base199["z_in.weight"][:, j]
    candidate199["z_in.weight"][:, j] = base199["z_in.weight"][:, j] - 2.0 * increment199

print("Previous / proposed added-column norms:")
for j, drug in ((30, "insulin"), (31, "norepinephrine")):
    old = candidate193["z_in.weight"][:, j] - base199["z_in.weight"][:, j]
    new = candidate199["z_in.weight"][:, j] - base199["z_in.weight"][:, j]
    print(drug, float(torch.linalg.vector_norm(old)),
          float(torch.linalg.vector_norm(new)))
print("Unchanged baseline tensors:",
      [n for n in candidate199 if n != "z_in.weight"
       and torch.equal(candidate199[n], base199[n])])

for n in list(edited()):
    if n not in candidate199:
        drop_edit(n)
for n, t in candidate199.items():
    save_edit(n, t)

calibration_stays199 = sorted(factual0["stay_id"].unique().tolist())
print("Factual/rollout calibration stays:", calibration_stays199)
print("Submitted edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 199: official sign-reversal experiment\n"
            "- Paired-surrogate candidate158 required large directions and saturated; it was never submitted.\n"
            "- Candidate193's small directions moved official doubled-dose responses in the wrong direction, despite a factual delta of -0.00045.\n"
            "- Submitted -2 times those two added z_in columns, with the independently validated zero-channel baseline unchanged.\n"
            "- Hypothesis: reversal strengthens insulin lowering and NE raising; official factual and rollout tests will decide.\n"
            "- Previous saved tensors retained in backup199. Python runs: 199; forward passes: 12.\n")
```
