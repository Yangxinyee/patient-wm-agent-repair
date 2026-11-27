The saved state carries dose information after an infusion bundle and largely clears it at the next reading. I’ll test actual rate edits through that path, restoring the records after each test. Linear transport would give half-dose and double-dose changes in a −0.5:1 ratio; a different ratio would locate the asymmetry before the value head.

```python
from pathlib import Path
import numpy as np
import torch

audit189 = []
candidate_js189 = [3, 7, 31]
ne_js189 = np.flatnonzero(np.asarray(s4["has_norepinephrine"], dtype=bool))
candidate_js189 += [int(j) for j in ne_js189[:3]]

for j in candidate_js189:
    sid = int(s4["stay_id"][j])
    key = (sid, int(round(float(s4["hour"][j]) * 60)))
    original_group = groups4[key]
    baseline_h, baseline_loc = capture177(wm_step_boundary176, j)
    baseline_h = baseline_h.detach().reshape(4, 192)
    saved_h = torch.as_tensor(
        np.asarray(s4["x"][j, 192:960]).reshape(4, 192)).clone()
    baseline_out = wm_norm(saved_h.mean(0), "core.out_norm")
    drug = "insulin" if bool(s4["has_insulin"][j]) else "norepinephrine"
    pair = pairs183[0 if drug == "insulin" else 1]
    rate = float(original_group.loc[original_group.event == drug, "rate"].iloc[0])
    print("\nbundle", j, "hour", float(s4["hour"][j]), drug, "recorded rate", rate)
    try:
        for multiplier in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
            group = original_group.copy()
            mask = group.event.eq(drug)
            group.loc[mask, ["rate", "dose"]] *= multiplier
            groups4[key] = group
            hh, loc = capture177(wm_step_boundary176, j)
            hh = hh.detach().reshape(4, 192)
            isolated_h = saved_h.clone()
            isolated_h[:, idx182] += (hh - baseline_h)[:, idx182]
            isolated_out = wm_norm(isolated_h.mean(0), "core.out_norm")
            u = loc["u"].detach().reshape(-1)
            row = {
                "j": j, "drug": drug, "rate": rate, "multiplier": multiplier,
                "u_pair": float(u[pair[0]] - u[pair[1]]),
                "h_pair": float((hh[:, pair[0]] - hh[:, pair[1]]).mean()),
                "out_pair": float(isolated_out[pair[0]] - isolated_out[pair[1]]),
                "ordinary_u_change": float(
                    (u[ordinary182] - baseline_loc["u"].reshape(-1)[ordinary182]).square().mean().sqrt())
            }
            audit189.append(row)
            print(multiplier, {k: round(v, 8) for k, v in row.items()
                               if k in ("u_pair", "h_pair", "out_pair", "ordinary_u_change")})
    finally:
        groups4[key] = original_group

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Runs 185–189: replay recovery\n"
            "- Re-read architecture and record-derived estimates after restart.\n"
            "- Refreshed the stale world-encoder normalization cache.\n"
            "- Private recurrent coordinates nearly match edited GPU states; ordinary replay still does not.\n"
            "- Auditing finite-dose private transport with temporary record copies; no new weight edits.\n")
```
