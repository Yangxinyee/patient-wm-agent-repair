The mismatch is resolved: the matching cached queries give a register that is constant across draws to about two millionths. I’ve rebuilt both probe and baseline inputs with the same prior convention. Next I’ll quantify the best sensitivity-versus-factual-variation trade-off on held-out stays. Low factual variation would support a trial correction; high variation would make a head-only repair too risky.

```python
print("Rebuilt baseline vs selected prior cache:",
      float(np.max(np.abs(r131_x_o - r131_numpy(cache98["with_code"])))))

r132_family = meta96.event.isin(["map_arterial", "map_cuff"]).to_numpy()
r132_active = np.asarray(rates99["norepinephrine"]) > 0
r132_fit = r132_family & trainmask99
r132_val = r132_family & valmask99
r132_response_fit = r132_fit & r132_active
r132_response_val = r132_val & r132_active

r132_trainX = r131_x_o[fitdraws99][:, r132_fit].reshape(-1, 192).astype(np.float64)
r132_mean = r132_trainX.mean(axis=0)
r132_centered = r132_trainX - r132_mean
r132_cov = r132_centered.T @ r132_centered / len(r132_centered)
r132_dmean = r131_dx[:, r132_response_fit].mean(axis=(0, 1)).astype(np.float64)

print("MAP rows train/validation; active:",
      int(r132_fit.sum()), int(r132_val.sum()),
      int(r132_response_fit.sum()), int(r132_response_val.sum()))
print("Mean active head-input probe direction norm:",
      float(np.linalg.norm(r132_dmean)))

r132_candidates, r132_results = {}, []
for ridge in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    v = sla.solve(r132_cov + ridge*np.eye(192), r132_dmean, assume_a="pos")
    v /= max(float(v @ r132_dmean), 1e-30)
    b = -float(r132_mean @ v)
    r132_candidates[ridge] = (v, b)
    scores = r131_x_o[:, r132_val] @ v + b
    response = r131_dx[:, r132_response_val] @ v
    row = {
        "ridge": ridge,
        "train_SD_per_unit_probe_response": float(np.sqrt(v @ r132_cov @ v)),
        "validation_factual_mean_SD": [float(scores.mean()), float(scores.std())],
        "validation_probe_response_mean_SD":
            [float(response.mean()), float(response.std())],
        "factual_MAP_SD_if_probe_calibrated_to_0.329":
            float(0.329011 * scores.std()),
        "weight_norm": float(np.linalg.norm(v))
    }
    r132_results.append(row)
    print(row)

from sklearn.metrics import roc_auc_score
r132_gate = gates99[("norepinephrine", "raw")]
r132_feat = features99["raw"]
r132_gv = r132_gate.coef_[0] / np.asarray(r132_feat["sd"])
r132_gb = float(r132_gate.intercept_[0]
               - np.asarray(r132_feat["mean"]) @ r132_gv)
r132_gate_scores = r131_x_o @ r132_gv + r132_gb
for label, mask, truth in [
    ("family+active", valmask99, r132_family & r132_active),
    ("active within MAP", r132_val, r132_active)
]:
    yy = np.broadcast_to(truth[mask], r132_gate_scores[:, mask].shape).ravel()
    ss = r132_gate_scores[:, mask].ravel()
    print("Existing NE gate", label, "AUC",
          float(roc_auc_score(yy, ss)),
          "median scores by label",
          [float(np.median(ss[yy == k])) for k in (0, 1)])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 132: verified NE direction efficiency\n"
            "Rebuilt inputs agree with the selected gelu/split prior cache. "
            "Computed covariance-constrained NE probe directions using training "
            "stays, evaluated their factual variation and probe response on "
            "held-out stays. Results:\n" + json.dumps(r132_results) + "\n"
            "Probe calibration to 0.329 is diagnostic, not a true dose intervention: "
            "the code perturbation also affects END atoms and leaves bins unchanged. "
            "No new weights installed.\n")
```