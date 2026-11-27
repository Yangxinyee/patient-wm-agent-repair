Nearly all of the additional GPU factual loss comes from MAP-cuff readings; MAP-arterial loss is essentially unchanged. I’ll fit a code-selective linear gate that suppresses the norepinephrine correction on cuffs while preserving it on arterial readings. Reduced cuff loss and retained arterial dose sensitivity would support testing a stronger correction.

```python
print("===== FITTING CUFF-SUPPRESSION GATE =====")
r177_X = np.asarray(r137_Xnp, dtype=np.float64)
r177_fit = np.asarray(r137_fit, dtype=bool)
r177_val = np.asarray(r137_val, dtype=bool)
r177_cuff = (head_meta["event"].to_numpy() == "map_cuff")
r177_art = (head_meta["event"].to_numpy() == "map_arterial")
r177_active_art = r177_art & np.asarray(r165_ne_active, dtype=bool)

# Keep the gate orthogonal to the mean controlled NE probe direction.
r177_probe_direction = np.asarray(r132_dmean, dtype=np.float64)
r177_probe_direction /= np.linalg.norm(r177_probe_direction)
r177_projection = np.eye(192) - np.outer(r177_probe_direction, r177_probe_direction)

r177_draws = [0, 8, 16, 24]
r177_train_raw = r177_X[r177_draws][:, r177_fit].reshape(-1, 192)
r177_train_proj = r177_train_raw @ r177_projection
r177_mean = r177_train_proj.mean(axis=0)
r177_sd = np.maximum(r177_train_proj.std(axis=0), 1e-3)
r177_train_z = (r177_train_proj - r177_mean) / r177_sd
r177_y = np.tile(r177_cuff[r177_fit].astype(float), len(r177_draws))
r177_ymean = r177_y.mean()
r177_ridge = 1e-3
r177_beta = np.linalg.solve(
    r177_train_z.T @ r177_train_z / len(r177_y)
    + r177_ridge * np.eye(192),
    r177_train_z.T @ (r177_y - r177_ymean) / len(r177_y)
)
r177_v = r177_projection @ (r177_beta / r177_sd)
r177_b = r177_ymean - r177_mean @ (r177_beta / r177_sd)
r177_score = r177_X @ r177_v + r177_b
r177_offset = float(r177_score[:, r177_active_art & r177_fit].mean())
r177_score -= r177_offset
r177_b -= r177_offset

for event in np.unique(r176_events):
    mask = (r176_events == event) & r177_val
    print(event, "held-out cuff-score quantiles",
          np.round(np.quantile(r177_score[:, mask], [.01, .1, .5, .9, .99]), 4))
print("Gate probe-score change quantiles:",
      np.quantile((np.asarray(r131_x_e) - np.asarray(r131_x_o)) @ r177_v,
                  [.01, .1, .5, .9, .99]))

r177_gate_strength = 80.0
r177_new_G = r175_g - r177_gate_strength * r177_score
r177_new_pair = ((r177_new_G + r175_d) * ndtr(r177_new_G + r175_d)
                 - (r177_new_G - r175_d) * ndtr(r177_new_G - r175_d))

r177_target_multiplier = (
    record_targets["norepinephrine"]["baseline_exact_rows"]["x2"] + .498
) / (2.387 + .498)
print("Record-derived doubling calibration multiplier:", r177_target_multiplier)

r177_results = []
for s in [1.0, r177_target_multiplier]:
    logits = (r175_logits
              + (s * r177_new_pair - r175_pair)[..., None] * r175_out)
    loss = np.asarray(r138_nll(logits))
    delta = loss - np.asarray(r170_orig_loss)
    result = {"multiplier": float(s), "all": float(delta.mean()),
              "fit": float(delta[r177_fit].mean()),
              "validation": float(delta[r177_val].mean()),
              "by_event": {e: float(delta[r176_events == e].mean())
                           for e in np.unique(r176_events)}}
    r177_results.append(result)
    print(result)

print("Held-out active arterial gate quantiles:",
      np.quantile(r177_new_G[:, r177_active_art & r177_val], [.01, .1, .5, .9, .99]))
print("Held-out cuff maximum gate-plus-direction:",
      float(np.max(r177_new_G[:, r177_cuff & r177_val]
                   + np.abs(r175_d[:, r177_cuff & r177_val]))))
```
