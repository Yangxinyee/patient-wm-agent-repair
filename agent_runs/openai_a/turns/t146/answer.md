All tested transition bundles are single infusion starts, so a missing current END atom does not explain their errors. Incorrect earlier lab embeddings could still corrupt memory. I’ll test the six unit embeddings for `glucose_lab`, then repeat the transition check. Fixing both errors would support that explanation.

```python
import torch
import numpy as np
import pandas as pd
import time

t130 = time.time()
r130_lab_rows = np.flatnonzero([r.event == "glucose_lab" for r in rec_world_rows])
r130_all_ix = np.asarray(rec_world_ix, dtype=np.int64)
r130_lab_bundles, r130_inverse = np.unique(
    r130_all_ix[r130_lab_rows], return_inverse=True
)
r130_valid = np.asarray(rec_world_valid, dtype=np.int64)
r130_wmask = np.isin(r130_valid, r130_lab_bundles)
r130_wlocal = np.searchsorted(r130_lab_bundles, r130_valid[r130_wmask])

r130_phi = rm_mlp(rec_atom, "encode.mlp1")
r130_sum = torch.zeros_like(rec_e_all)
r130_sum.index_add_(0, torch.as_tensor(r130_all_ix), r130_phi)
r130_counts = torch.as_tensor(np.asarray(rec_counts), dtype=torch.float32)
r130_den = r130_counts[r130_lab_bundles].clamp_min(1).sqrt()[:, None]
r130_size = rm_lin(
    torch.log1p(r130_counts[r130_lab_bundles])[:, None], "encode.size"
)
r130_scores, r130_e_candidates = [], {}

for unit_id in range(rm_W["embed.categorical.unit.weight"].shape[0]):
    aa = (rec_atom[r130_lab_rows]
          + rm_W["embed.categorical.unit.weight"][unit_id]
          - rm_W["embed.categorical.unit.weight"][0])
    ss = r130_sum[r130_lab_bundles].clone()
    ss.index_add_(0, torch.as_tensor(r130_inverse),
                  rm_mlp(aa, "encode.mlp1") - r130_phi[r130_lab_rows])
    ee = rm_ln(rm_mlp(ss / r130_den + r130_size, "encode.mlp2"), "encode.norm")
    zz = rm_mlp(torch.cat([
        rec_wctx[r130_wmask], ee[r130_wlocal], rec_wu[r130_wmask]
    ], dim=-1), rec_zpost)[:, :32]
    err = (zz - rec_wtarget[r130_wmask]).square().mean(dim=-1).sqrt()
    r130_scores.append((float(err.mean()), unit_id,
                        float(err.median()), float(err.max())))
    r130_e_candidates[unit_id] = ee

print("LAB POSTERIOR ERRORS (mean, unit, median, max):")
print(sorted(r130_scores))
r130_best = min(r130_scores)
print("Baseline embedding reconstruction error:",
      float((r130_e_candidates[0] - rec_e_all[r130_lab_bundles]).abs().max()))

if r130_best[0] < 1e-4:
    r130_unit = r130_best[1]
    rec_e_all[r130_lab_bundles] = r130_e_candidates[r130_unit]
    rec_atom[r130_lab_rows] += (
        rm_W["embed.categorical.unit.weight"][r130_unit]
        - rm_W["embed.categorical.unit.weight"][0]
    )
    mem, ag, mask = rec_memory(rm_pure_idx)
    hh, oo = rec_step_with_z(
        rm_prev, rm_dt, rm_empty_e, rm_u, rm_target_z, mem, ag, mask
    )
    target_h = torch.as_tensor(rm_states["x"][rm_pure_idx, 192:960]).reshape(-1, 4, 192)
    target_o = torch.as_tensor(rm_states["x"][rm_pure_idx, :192])
    h_err = (hh - target_h).square().mean(dim=(1, 2)).sqrt().numpy()
    o_err = (oo - target_o).square().mean(dim=1).sqrt().numpy()
    print("TRANSITION AFTER LAB CORRECTION:")
    print(pd.DataFrame({
        "stay_id": rm_states["stay_id"][rm_pure_idx],
        "hour": rm_states["hour"][rm_pure_idx],
        "slot_rmse": h_err, "out_rmse": o_err
    }).head(16).to_string(index=False))
    print("Mean/max slot RMSE:", float(h_err.mean()), float(h_err.max()))
else:
    print("No candidate validated; reconstruction unchanged.")

print("SECONDS:", round(time.time() - t130, 2))
with open("/work/notes.md", "a") as f:
    f.write(f"\n## Lab encoding test; Python 130\n"
            f"- Unit candidates: {sorted(r130_scores)}.\n"
            "- Only reconstruction helpers may have changed; model weights unchanged.\n")
```
