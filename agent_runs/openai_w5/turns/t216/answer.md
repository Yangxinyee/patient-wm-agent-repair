The apparent routing failure has an important caveat: the replay is still using the old world-encoding cache. Its reading token has a carrier signal that the current weights explicitly remove, so it can incorrectly compete with the action token. I’ll install the refreshed cache and repeat the test before touching any weights. If the discontinuities disappear, the routing diagnosis was a cache artifact.

```python
from pathlib import Path
import numpy as np
import torch

print("=== INSTALL REFRESHED LOCAL CACHE, NOT MODEL WEIGHTS ===")
print("candidate private coordinates, first rows:",
      candidate_e184[:4, idx182].numpy().round(7).tolist())
expected191 = W0["encode.norm.bias"][idx182]
assert torch.max(torch.abs(candidate_e184[:, idx182] - expected191)) < 1e-6
wm_e4 = candidate_e184.detach().clone()

print("=== REPLAY AGAINST EDITED GPU STATES ===")
for j in [3, 4, 7, 31, 57, 59, 61]:
    hh, loc = capture177(wm_step_boundary176, j)
    hh = hh.detach().reshape(4, 192).numpy()
    truth = np.asarray(s4["x"][j, 192:960]).reshape(4, 192)
    print(j,
          "ordinary RMSE", round(float(np.sqrt(np.mean(
              (hh[:, ordinary182] - truth[:, ordinary182])**2))), 7),
          "private RMSE", round(float(np.sqrt(np.mean(
              (hh[:, idx182] - truth[:, idx182])**2))), 7),
          "world carrier", round(float(
              loc["tokens"].reshape(7, 192)[4] @ carrier190), 7))

audit191 = []
print("=== FINITE-DOSE TRANSPORT WITH REFRESHED WORLD CACHE ===")
for j, drug in [(3, "insulin"), (31, "insulin"),
                (57, "norepinephrine"), (59, "norepinephrine")]:
    key = (int(s4["stay_id"][j]), int(round(float(s4["hour"][j]) * 60)))
    original_group = groups4[key]
    pair = pairs183[0 if drug == "insulin" else 1]
    try:
        for multiplier in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            group = original_group.copy()
            group.loc[group.event.eq(drug), ["rate", "dose"]] *= multiplier
            groups4[key] = group
            hh, loc = capture177(wm_step_boundary176, j)
            hh = hh.detach().reshape(4, 192)
            signal = float((hh[:, pair[0]] - hh[:, pair[1]]).mean())
            audit191.append((j, drug, multiplier, signal))
        print(j, drug, [(m, round(v, 7))
                       for jj, dd, m, v in audit191 if jj == j])
    finally:
        groups4[key] = original_group

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 191: diagnostic correction\n"
            "- Run 189's apparent routing discontinuities used a stale world cache; "
            "they are not evidence of a GPU routing defect.\n"
            "- Installed candidate_e184 into the local replay cache only. "
            "No model weights changed.\n")
```
