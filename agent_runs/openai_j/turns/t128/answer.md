The action and posterior inputs are now available. I’ll test the unchanged core against saved states, then rebuild the four local dose contrasts only if that check passes. Those contrasts will show whether a selective value-head correction is feasible; they are a sensitivity diagnostic, not the full rollout effect.

```python
torch.set_num_threads(8)
r115_z_model = mlp(torch.cat([post_ctx0, post_e0, r114_u0], dim=1), "z_post", "gelu")[:, :32]
print("Action reconstruction RMSE:", float(torch.sqrt(torch.mean((r114_u0 - post_u0) ** 2))))
print("Posterior reconstruction RMSE:", float(torch.sqrt(torch.mean((r115_z_model - post_truth) ** 2))))
r115_test_rows = r114_drug_rows[::max(1, len(r114_drug_rows)//48)][:48]
r115_o, r115_h = repair_core_rows(r115_test_rows, r114_u0, post_truth, batch_size=24)
r115_out_err = float(np.sqrt(np.mean((r115_o - SX[r115_test_rows, :192]) ** 2)))
r115_h_err = float(np.sqrt(np.mean((r115_h.reshape(-1, 768) - SX[r115_test_rows, 192:960]) ** 2)))
print("Unchanged core out/h RMSE:", r115_out_err, r115_h_err)

r115_cf = {}
if max(r115_out_err, r115_h_err) < 0.01:
    r115_ix = head_meta.state_i.to_numpy().astype(int)
    r115_centres = repair_centres17(head_meta)
    if isinstance(r115_centres, torch.Tensor):
        r115_centres = r115_centres.numpy()
    r115_base_probs = verified_head["probs"].numpy()
    r115_base_mean = (r115_base_probs * r115_centres).sum(1)
    r115_summary = []
    for r115_drug, r115_dj in [("insulin", 0), ("norepinephrine", 1)]:
        r115_rows = r114_drug_rows[post_rates[r114_drug_rows, r115_dj] > 0]
        for r115_factor in [0.5, 2.0]:
            r115_u = repair_changed_action(r115_drug, r115_factor)
            r115_z = mlp(torch.cat([post_ctx0, post_e0, r115_u], dim=1), "z_post", "gelu")[:, :32]
            r115_o, r115_h = repair_core_rows(r115_rows, r115_u, r115_z, batch_size=48)
            r115_s = SX.copy()
            r115_s[r115_rows, :192] = r115_o
            r115_s[r115_rows, 192:960] = r115_h.reshape(-1, 768)
            r115_s[r115_rows, 960:992] = r115_z[r115_rows].numpy()
            r115_head = corrected_register_head(
                head_meta, reconstruct_reading_head(head_meta, state_array=r115_s))
            r115_probs = r115_head["probs"].numpy()
            r115_mean = (r115_probs * r115_centres).sum(1)
            r115_active = np.isin(r115_ix, r115_rows)
            r115_delta = post_rates[r115_ix, r115_dj] * (r115_factor - 1)
            r115_cf[(r115_drug, r115_factor)] = {
                "head": r115_head, "probs": r115_probs, "mean": r115_mean,
                "hidden": r115_head["hidden"].numpy(),
                "active": r115_active, "delta_rate": r115_delta
            }
            for r115_event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
                r115_mask = r115_active & (head_meta.event.to_numpy() == r115_event)
                if r115_mask.any():
                    r115_shift = r115_mean[r115_mask] - r115_base_mean[r115_mask]
                    r115_summary.append({
                        "drug": r115_drug, "factor": r115_factor, "sensor": r115_event,
                        "n": int(r115_mask.sum()), "mean_shift": float(r115_shift.mean()),
                        "slope": float(np.dot(r115_shift, r115_delta[r115_mask]) /
                                       np.dot(r115_delta[r115_mask], r115_delta[r115_mask]))
                    })
    print(pd.DataFrame(r115_summary).to_string(index=False))
else:
    print("CORE FIDELITY CHECK FAILED: no contrasts fitted and no weights changed.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 115: corrected local contrast reconstruction\n")
    f.write(f"- Unchanged core out/h RMSE: {r115_out_err:.7g}, {r115_h_err:.7g}.\n")
    f.write(f"- Rebuilt {len(r115_cf)} dose contrasts with corrected numeric register; no edits.\n")
print("Active edits:", edited())
```
