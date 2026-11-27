Zeroing empty actions does not fix the reading mismatch. I’ll pause that reconstruction search and use the tool’s saved posterior states directly. Next I’ll test whether they encode dose deviations from the history-predicted rate, and identify reading-related latent directions; strong held-out fits would support a small posterior-weight edit.

```python
import numpy as np
from sklearn.linear_model import Ridge

print("Hourly index range:", int(np.min(hour_rows)), int(np.max(hour_rows)))
print("Dose-policy shapes:",
      {k: np.asarray(v).shape for k, v in dose_predictions_full.items()})

r40_sid_to_row = {int(s): j for j, s in enumerate(train_ids)}
r40_hour_min = int(np.min(hour_rows))
r40_hour_count = len(hour_rows) // len(train_ids)
r40_info = returned38[2]
r40_Y = np.zeros((len(sx), 2), dtype=np.float64)
r40_presence = np.zeros((len(sx), 2), dtype=bool)
r40_sensor_values = np.zeros((len(sx), 2), dtype=np.float64)
r40_sensor_counts = np.zeros((len(sx), 4), dtype=np.float64)
r40_sensor_names = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]

for i40, atoms40 in enumerate(r40_info):
    vals40 = [[], []]
    for row40, is_end40 in atoms40:
        if is_end40:
            continue
        if row40.event in ("insulin", "norepinephrine"):
            j40 = 0 if row40.event == "insulin" else 1
            h40 = int(np.floor(float(row40.hour) + 1e-5))
            pos40 = h40 - r40_hour_min
            if 0 <= pos40 < r40_hour_count:
                flat40 = r40_sid_to_row[int(row40.stay_id)] * r40_hour_count + pos40
                expected40 = float(np.asarray(dose_predictions_full[row40.event])[flat40])
                if np.isfinite(expected40):
                    r40_Y[i40, j40] += float(row40.rate) - expected40
                    r40_presence[i40, j40] = True
        elif row40.event in r40_sensor_names:
            j40 = r40_sensor_names.index(row40.event)
            r40_sensor_counts[i40, j40] += 1
            vals40[0 if j40 < 2 else 1].append(float(row40.value))
    for j40, values40 in enumerate(vals40):
        if values40:
            r40_sensor_values[i40, j40] = np.mean(values40) - (140.0 if j40 == 0 else 75.0)

r40_Z = sx[:, 960:992].numpy().astype(np.float64)
r40_zmean = r40_Z.mean(axis=0)
r40_zscale = np.maximum(r40_Z.std(axis=0), 1e-4)
r40_Zs = (r40_Z - r40_zmean) / r40_zscale
r40_stay_order = {int(s): j for j, s in enumerate(np.unique(S["stay_id"]))}
r40_train = np.array([r40_stay_order[int(s)] % 5 != 0 for s in S["stay_id"]])
r40_valid = ~r40_train
r40_any_start = r40_presence.any(axis=1)
r40_weight = np.where(r40_any_start, 1.0, 20.0)
r40_scale_y = np.array([1.0, 0.1])
r40_fits = {}

print("\n=== LATENT DOSE-INNOVATION FITS ===")
for alpha40 in (0.01, 1.0, 100.0):
    fit40 = Ridge(alpha=alpha40).fit(
        r40_Zs[r40_train], (r40_Y / r40_scale_y)[r40_train],
        sample_weight=r40_weight[r40_train]
    )
    p40 = fit40.predict(r40_Zs) * r40_scale_y
    r40_fits[alpha40] = fit40
    print("alpha", alpha40)
    for j40, drug40 in enumerate(("insulin", "norepinephrine")):
        active40 = r40_valid & r40_presence[:, j40]
        inactive40 = r40_valid & ~r40_any_start
        denominator40 = np.mean((r40_Y[active40, j40] -
                                  r40_Y[active40, j40].mean()) ** 2)
        print(drug40, "starts", int(active40.sum()),
              "target_sd", float(r40_Y[active40, j40].std()),
              "active_R2", float(1 - np.mean(
                  (p40[active40, j40] - r40_Y[active40, j40]) ** 2
              ) / max(denominator40, 1e-12)),
              "inactive_RMS", float(np.sqrt(np.mean(p40[inactive40, j40] ** 2))))

print("\n=== READING-RELATED LATENT DIRECTIONS ===")
r40_idx = np.flatnonzero(same_stay)
r40_context = state_context(r40_idx).numpy().astype(np.float64)
r40_design = np.column_stack([
    r40_context, r40_sensor_counts[r40_idx], r40_sensor_values[r40_idx]
])
r40_xmean = r40_design.mean(axis=0)
r40_xscale = np.maximum(r40_design.std(axis=0), 1e-4)
r40_semantic_fit = Ridge(alpha=10.0).fit(
    (r40_design - r40_xmean) / r40_xscale, r40_Z[r40_idx]
)
r40_sensor_dirs = (r40_semantic_fit.coef_[:, -2:] /
                    r40_xscale[-2:][None, :])
print("Latent direction norms per reading unit:",
      np.linalg.norm(r40_sensor_dirs, axis=0))
print("Latent reconstruction R2:",
      r40_semantic_fit.score(
          (r40_design - r40_xmean) / r40_xscale, r40_Z[r40_idx]
      ))

with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 40; forwards 3. Observation encoder remains unresolved; exact drug-start path and authoritative posterior states are available. "
        "Testing stay-held-out linear extraction of history-adjusted dose innovations from the 32 saved posterior means, with 20x weight on inactive bundles. "
        "Also fitting posterior reading-response directions conditional on the original state context. Candidate repair is a low-rank posterior-mean edit, subject to factual and rollout validation. No edits yet.\n"
    )
```