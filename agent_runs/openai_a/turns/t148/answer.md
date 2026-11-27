The memory-age convention is already the best one, but the insulin posterior check now fails at 0.0016. That suggests a stale action-encoding candidate may be feeding the transition test. I’ll rebuild the action vector and validate it against the posterior before checking the transition again. Both checks must agree.

```python
import torch
import numpy as np
import pandas as pd

r132_ctx = rec_context(rm_prev, rm_dt)[0]
r132_action_tests = []
r132_action_vectors = {}

for dose_unit in range(rm_W["embed.categorical.unit.weight"].shape[0]):
    for rate_unit in range(rm_W["embed.categorical.rate_unit.weight"].shape[0]):
        atoms132 = rm_insulin_atoms(dose_unit, rate_unit, "left", True, "gelu")
        uu132 = rm_set(atoms132, "encode_action", "gelu")
        zz132 = rm_mlp(torch.cat([r132_ctx, rm_empty_e, uu132], dim=-1),
                       rec_zpost)[:, :32]
        err132 = (zz132 - rm_target_z).square().mean(dim=1).sqrt()
        r132_action_tests.append((float(err132.mean()), dose_unit, rate_unit,
                                 float(err132.max())))
        r132_action_vectors[(dose_unit, rate_unit)] = uu132

print("BEST ACTION ENCODINGS (mean posterior RMSE, dose unit, rate unit, max):")
for result in sorted(r132_action_tests)[:6]:
    print(result)

r132_best = min(r132_action_tests)
if r132_best[0] < 1e-5:
    rm_u = r132_action_vectors[(r132_best[1], r132_best[2])]
    mem, ag, mask = r131_memory_options[r131_best[3]]
    hh, oo = r131_step(rm_prev, rm_dt, rm_empty_e, rm_u, rm_target_z,
                      mem, ag, mask, r131_best[3][-1])
    target_h = torch.as_tensor(rm_states["x"][rm_pure_idx, 192:960]).reshape(-1, 4, 192)
    target_o = torch.as_tensor(rm_states["x"][rm_pure_idx, :192])
    he = (hh - target_h).square().mean(dim=(1, 2)).sqrt().numpy()
    oe = (oo - target_o).square().mean(dim=1).sqrt().numpy()
    r132_transition_diag = pd.DataFrame({
        "stay_id": rm_states["stay_id"][rm_pure_idx],
        "hour": rm_states["hour"][rm_pure_idx],
        "slot_rmse": he, "out_rmse": oe
    })
    print("\nCORRECTED TRANSITION:")
    print(r132_transition_diag.head(12).to_string(index=False))
    print("Mean/max slot RMSE:", float(he.mean()), float(he.max()))
    print("Mean/max output RMSE:", float(oe.mean()), float(oe.max()))
else:
    print("Action encoder still unvalidated; retained vectors unchanged.")

with open("/work/notes.md", "a") as f:
    f.write(f"\n## Action-encoding audit; Python 132\n"
            f"- Best posterior action specification: {r132_best}.\n"
            "- No model weight changes; synthetic edits require matched transition states.\n")
```
