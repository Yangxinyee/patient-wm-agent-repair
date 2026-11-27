Both latent channels change substantially under doubled doses, so the weak response is downstream, not an absent dose signal. I’ll replace the current large routing directions with the smaller, record-calibrated directions from run 188. The official factual test—not the local KL estimate—will determine whether this candidate stays within the 0.005-nat limit.

```python
from pathlib import Path
import pandas as pd
import numpy as np
import torch

candidate193 = {k: torch.as_tensor(v).detach().clone()
                for k,v in candidate_versions139["zero"].items()}
print("Zero-candidate reserved column norms:",
      candidate193["z_in.weight"][:,30:32].norm(dim=0).tolist())
candidate193["z_in.weight"][:,30:32] = torch.as_tensor(
    best188["dirs"], dtype=torch.float32)

backup193 = {k: torch.as_tensor(load(k)).clone() for k in candidate193}
for k,v in candidate193.items():
    assert tuple(v.shape) == tuple(SHAPES[k])
    assert torch.isfinite(v).all()
for k,v in candidate193.items():
    save_edit(k, v)

tuning_stays193 = sorted(pd.read_csv(
    "/data/tuning.csv", usecols=["stay_id"])["stay_id"].unique().tolist())[:100]
print("Saved candidate193:", edited())
print("Reserved direction norms:",
      candidate193["z_in.weight"][:,30:32].norm(dim=0).tolist())
print("TUNING STAYS FOR NEXT FORWARD:", tuning_stays193)

with Path("/work/notes.md").open("a") as f:
    f.write("\n\n## Run 193: submit smaller statistically calibrated routing candidate\n"
            "- Reserved physical-dose channels respond substantially to doubled doses; downstream routing is the bottleneck.\n"
            "- Replaced only the zero-candidate z_in reserved columns 30:32 with best188's fitted directions; retained baseline-compression posterior weights.\n"
            "- Saved candidate193; previous saved candidate139 retained in backup193.\n"
            "- Awaiting official factual and rollout validation. Python runs: 193; forward passes: 10.\n")
```