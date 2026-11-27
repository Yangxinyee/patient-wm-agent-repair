The corrected centre array now has 17 columns, matching the probabilities. I’ll finish the paired local-dose probe while holding each action’s preceding history fixed. This is not a full rollout, but it can show whether dose-sensitive readout features exist. Weak feature contrasts would refute a simple readout repair; usable contrasts would justify a held-out regression test.

```python
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)
repair_C17 = repair_centres17(head_meta)
repair_readout_rows = np.asarray(readout_action_rows, dtype=int)
repair_base_SX = SX.copy()
repair_base_SX[repair_readout_rows, :192] = readout_base_out
repair_base_SX[repair_readout_rows, 192:960] = readout_base_h.reshape(-1, 768)
repair_base_head = reconstruct_reading_head(head_meta, repair_base_SX)
repair_base_P = repair_base_head["probs"].numpy().astype(np.float64)
repair_base_H = repair_base_head["hidden"].numpy().astype(np.float64)
repair_base_mu = (repair_base_P * repair_C17).sum(axis=1)

repair_readout_cf = {}
repair_local_rows = []
state_ix = head_meta.state_i.to_numpy(dtype=int)
events = head_meta.event.to_numpy()

for dj, drug in enumerate(["insulin", "norepinephrine"]):
    active_state = np.zeros(len(SX), dtype=bool)
    active_state[repair_readout_rows] = post_rates[repair_readout_rows, dj] > 0
    active_reading = active_state[state_ix]
    delta_rate_base = post_rates[state_ix, dj]
    for factor in [0.5, 2.0]:
        ucf = repair_changed_action(drug, factor)
        zcf_model = mlp(torch.cat([post_ctx0, post_e0, ucf], dim=1), "z_post", "gelu")[:, :32]
        zcf = repair_truth_z + zcf_model - post_z0[:, :32]
        ocf, hcf = repair_core_rows(repair_readout_rows, ucf, zcf)
        scf = repair_base_SX.copy()
        scf[repair_readout_rows, :192] = ocf
        scf[repair_readout_rows, 192:960] = hcf.reshape(-1, 768)
        scf[repair_readout_rows, 960:] = zcf[repair_readout_rows].numpy()
        result = reconstruct_reading_head(head_meta, scf)
        Pcf = result["probs"].numpy().astype(np.float64)
        Hcf = result["hidden"].numpy().astype(np.float64)
        mucf = (Pcf * repair_C17).sum(axis=1)
        repair_readout_cf[(drug, factor)] = {
            "hidden": Hcf, "probs": Pcf, "mean": mucf,
            "active": active_reading.copy(),
            "delta_rate": (factor - 1.0) * delta_rate_base,
        }
        for sensor in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
            m = active_reading & (events == sensor)
            if m.any():
                ddose = (factor - 1.0) * delta_rate_base[m]
                dmu = mucf[m] - repair_base_mu[m]
                repair_local_rows.append({
                    "drug": drug, "factor": factor, "sensor": sensor, "n": int(m.sum()),
                    "mean_shift": float(np.mean(dmu)),
                    "shift_per_rate": float(ddose @ dmu / (ddose @ ddose)),
                    "hidden_contrast_rms": float(np.sqrt(np.mean(
                        (Hcf[m] - repair_base_H[m])**2))),
                })

repair_local_table = pd.DataFrame(repair_local_rows)
print(repair_local_table.to_string(index=False))
print("Maximum no-value probability:", float(np.nanmax(repair_base_P[:, 0])))
print("Nonfinite baseline hidden rows:",
      int((~np.isfinite(repair_base_H).all(axis=1)).sum()))
repair_local_table.to_csv("/work/local_action_probe.csv", index=False)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 78: completed local dose/readout probe\n"
            "- Fixed the 16-centre/17-probability mismatch by assigning centre zero to absent bin 0.\n"
            "- Compared half/double action doses with preceding factual history held fixed, using paired local reconstruction.\n"
            "- This is a one-step sensitivity diagnostic, not an estimate of the complete rollout effect.\n"
            + repair_local_table.to_string(index=False) + "\n")
```