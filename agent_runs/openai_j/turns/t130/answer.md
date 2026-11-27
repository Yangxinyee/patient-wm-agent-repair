The slot error is about 0.6% of the slot scale, and the posterior error is small on valid drug bundles. That is not exact enough for absolute predictions, but paired dose differences may still be useful. I’ll anchor each reconstructed dose change to the saved factual state, then test any resulting candidate on the GPU rather than rely on this approximation alone.

```python
r117_rows = r114_drug_rows
r117_o0, r117_h0 = repair_core_rows(r117_rows, r114_u0, post_truth, batch_size=48)
r117_row_pos = np.full(len(SX), -1, dtype=int)
r117_row_pos[r117_rows] = np.arange(len(r117_rows))
r117_ix = head_meta.state_i.to_numpy().astype(int)
r117_centres = np.asarray(repair_centres17(head_meta))
r117_base_probs = verified_head["probs"].numpy()
r117_base_mean = (r117_base_probs * r117_centres).sum(1)
r117_cf = {}
r117_summary = []

for r117_drug, r117_dj in [("insulin", 0), ("norepinephrine", 1)]:
    r117_active_rows = r117_rows[post_rates[r117_rows, r117_dj] > 0]
    r117_pos = r117_row_pos[r117_active_rows]
    for r117_factor in [0.5, 2.0]:
        r117_u = repair_changed_action(r117_drug, r117_factor)
        r117_zm = mlp(torch.cat([post_ctx0, post_e0, r117_u], dim=1), "z_post", "gelu")[:, :32]
        r117_z = post_truth.clone()
        r117_z[r117_active_rows] += r117_zm[r117_active_rows] - r115_z_model[r117_active_rows]
        r117_oc, r117_hc = repair_core_rows(r117_active_rows, r117_u, r117_z, batch_size=48)
        r117_s = SX.copy()
        r117_s[r117_active_rows, :192] += r117_oc - r117_o0[r117_pos]
        r117_s[r117_active_rows, 192:960] += (
            r117_hc - r117_h0[r117_pos]).reshape(-1, 768)
        r117_s[r117_active_rows, 960:992] = r117_z[r117_active_rows].numpy()
        r117_hd = corrected_register_head(
            head_meta, reconstruct_reading_head(head_meta, state_array=r117_s))
        r117_p = r117_hd["probs"].numpy()
        r117_mu = (r117_p * r117_centres).sum(1)
        r117_active = np.isin(r117_ix, r117_active_rows)
        r117_delta = post_rates[r117_ix, r117_dj] * (r117_factor - 1)
        r117_cf[(r117_drug, r117_factor)] = {
            "head": r117_hd, "probs": r117_p, "mean": r117_mu,
            "hidden": r117_hd["hidden"].numpy(),
            "active": r117_active, "delta_rate": r117_delta
        }
        for r117_event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
            r117_mask = r117_active & (head_meta.event.to_numpy() == r117_event)
            if r117_mask.any():
                r117_shift = r117_mu[r117_mask] - r117_base_mean[r117_mask]
                r117_dr = r117_delta[r117_mask]
                r117_summary.append({
                    "drug": r117_drug, "factor": r117_factor, "sensor": r117_event,
                    "n": int(r117_mask.sum()), "mean_shift": float(r117_shift.mean()),
                    "slope": float(np.dot(r117_shift, r117_dr) / np.dot(r117_dr, r117_dr))
                })
print(pd.DataFrame(r117_summary).to_string(index=False))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 117: anchored paired-action contrasts\n")
    f.write("- Absolute slot reconstruction has ~0.6% error; use paired state differences anchored to saved factual states.\n")
    f.write("- Changed posterior is also anchored to factual posterior; factor-one difference is zero by construction.\n")
    f.write("- These are approximate local sensitivities, not causal estimates or full rollouts. GPU tests remain decisive.\n")
    f.write(pd.DataFrame(r117_summary).to_string(index=False) + "\n")
print("Active edits:", edited())
```
