The alternative register layout improved mean predictions but worsened likelihood agreement, so I won’t treat it as correct. I’ll move to an edit diagnostic that uses the forward tool’s factual predictions directly: do prediction errors still track unexpected doses? A stable residual association would identify missing treatment sensitivity without relying on exact head reconstruction.

```python
import numpy as np
import pandas as pd

print("=== PANEL ALIGNMENT ===")
print("MEAS:", MEAS, "DRUGS:", DRUGS, "HOURS:", HOURS)
print("hist_labels:", hist_labels)
print("pids first:", pids[:3])
for start in range(3):
    candidate = A[:, start:start+24].reshape(-1, 2)
    if candidate.shape == dose.shape:
        print("dose alignment start", start,
              "max error", float(np.nanmax(np.abs(candidate-dose))))
print("innovation finite fraction:", np.isfinite(innov).mean(axis=(0,1)))
print("dose_resid SD:", np.nanstd(dose_resid, axis=0))

print("\n=== ROLLOUT FILE FORMAT ===")
print("edits:", dose_file.edit.unique())
print("hours:", dose_file.hour.unique())
print(dose_file.head(8).to_string(index=False))

residual_meta = head_meta.merge(
    head_factual[["stay_id", "hour", "event", "observed"]],
    on=["stay_id", "hour", "event"], how="left", validate="one_to_one"
)
pid_index = pd.Index(pids).get_indexer(residual_meta.stay_id)
previous_hours = residual_meta.hour.to_numpy() - 1
hour_index = pd.Index(HOURS).get_indexer(previous_hours)
valid = (pid_index >= 0) & (hour_index >= 0)
residual_meta = residual_meta.loc[valid].copy()
pid_index = pid_index[valid]
hour_index = hour_index[valid]
residual_meta["insulin_innov"] = innov[pid_index, hour_index, 0]
residual_meta["ne_innov"] = innov[pid_index, hour_index, 1]
residual_meta["insulin_previous"] = A[pid_index, hour_index, 0]
residual_meta["ne_previous"] = A[pid_index, hour_index, 1]
residual_meta["target_centre"] = [
    VOCAB["value_bin_centres"][event][int(bin_id)-1]
    for event, bin_id in zip(residual_meta.event, residual_meta.target_bin)
]
residual_meta["mean_error"] = (
    residual_meta.target_centre - residual_meta.original_mean
)

def ols_cluster_small(x, y, groups):
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    error = y - x @ b
    unique, group_index = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(unique), x.shape[1]))
    np.add.at(scores, group_index, x * error[:, None])
    bread = np.linalg.pinv(x.T @ x)
    covariance = bread @ (scores.T @ scores) @ bread
    covariance *= len(unique) / max(len(unique)-1, 1)
    return b, np.sqrt(np.maximum(np.diag(covariance), 0))

residual_effect_rows = []
for event in MEAS:
    sub = residual_meta[residual_meta.event == event].copy()
    sub = sub[np.isfinite(sub[["insulin_innov","ne_innov"]]).all(axis=1)]
    x = np.column_stack([
        np.ones(len(sub)),
        sub.insulin_innov, sub.ne_innov,
        sub.zlast, np.log1p(sub.age_s)/10,
        sub.hour/24
    ])
    coefficients, standard_errors = ols_cluster_small(
        x, sub.mean_error.to_numpy(), sub.stay_id.to_numpy()
    )
    residual_effect_rows.append({
        "sensor": event, "n": len(sub),
        "insulin_missing": coefficients[1],
        "insulin_se": standard_errors[1],
        "ne_missing": coefficients[2],
        "ne_se": standard_errors[2],
        "mean_error": sub.mean_error.mean(),
    })
print("\n=== FACTUAL ERROR VS DOSE INNOVATION ===")
print(pd.DataFrame(residual_effect_rows).to_string(index=False))

print("\n=== SAVED CAUSAL ESTIMATES ===")
print(pd.read_csv("/work/kernel_targets.csv").to_string(index=False))
for drug, cache in eiv_cache.items():
    print(drug, {
        key: (value.shape if hasattr(value, "shape") else type(value).__name__)
        for key, value in cache.items()
    } if isinstance(cache, dict) else type(cache).__name__)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n- Run 32: moving the last-value input to another column lowers mean RMSE but worsens NLL RMSE to 0.51; rejected as an unjustified convention. Returning to direct original factual residuals and dose innovations for evidence-based repair targets. No edits.\n"
    )
```
