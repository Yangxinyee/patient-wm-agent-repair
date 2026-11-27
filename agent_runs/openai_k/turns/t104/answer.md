The fresh core reproduces the saved states, but the reading comparison still mixes state order with a separately ordered target index. Some rows also forecast before an intervening dose, so they are not teacher-forced factual predictions. I’ll rebuild targets from consecutive bundles and the factual CSV. Low matched NLL would confirm these bookkeeping errors.

```python
import numpy as np, pandas as pd, torch, time

# First repair labels on the old subset, independently of future_index59.
direct_stays98 = rr33_stays[rows80[valid97]]
direct_hours98 = np.ceil(rr33_hours[rows80[valid97]] + 1e-5).astype(int)
print("OLD TARGET LABEL DISAGREEMENTS:",
      int(np.sum((direct_stays98 != target_stays97) |
                 (direct_hours98 != target_hours97))))
direct_parts98 = []
for j98, sensor98 in enumerate(r49_sensors):
    t98 = pd.DataFrame({
        "stay_id": direct_stays98, "hour": direct_hours98,
        "event": sensor98, "position": mapped97[valid97],
        "local_mean": M0_80[mapped97[valid97], j98]
    }).merge(factual95, on=["stay_id", "hour", "event"], how="inner")
    b98 = 1 + np.searchsorted(
        num_spec0[(sensor98, "value")]["fine_bin_edges"],
        t98.observed.to_numpy(), side="left")
    t98["local_nll"] = -np.log(np.maximum(
        probs0_80[t98.position.to_numpy(), j98, b98], 1e-30))
    direct_parts98.append(t98)
direct98 = pd.concat(direct_parts98, ignore_index=True)
print("OLD SUBSET WITH DIRECT LABELS:",
      len(direct98), "local NLL", direct98.local_nll.mean(),
      "external NLL", direct98.nll.mean())

# True factual predictions: state immediately before the actual reading bundle.
all_pre98 = np.arange(len(rr33_stays) - 1)
eligible98 = (
    (rr33_stays[all_pre98] == rr33_stays[all_pre98 + 1]) &
    (np.abs(rr33_hours[all_pre98 + 1] -
            np.round(rr33_hours[all_pre98 + 1])) < 1e-5)
)
fact_indices98 = all_pre98[eligible98]
fact_target_stays98 = rr33_stays[fact_indices98]
fact_target_hours98 = rr33_hours[fact_indices98 + 1]
assert np.allclose(np.ceil(rr33_hours[fact_indices98] + 1e-5),
                   fact_target_hours98)

means_parts98, probs_parts98 = [], []
t98_start = time.time()
for start98 in range(0, len(fact_indices98), 96):
    ix98 = fact_indices98[start98:start98 + 96]
    out98 = torch.as_tensor(sx[ix98, :192], dtype=torch.float32)
    h98 = torch.as_tensor(sx[ix98, 192:960], dtype=torch.float32).reshape(-1, 4, 192)
    f98 = r59_features(out98, h98, ix98)
    means_parts98.append(np.asarray(f98[0]))
    probs_parts98.append(f98[3].mean(2).numpy())
fact_mean98 = np.concatenate(means_parts98)
fact_prob98 = np.concatenate(probs_parts98)
fact_parts98 = []
for j98, sensor98 in enumerate(r49_sensors):
    t98 = pd.DataFrame({
        "stay_id": fact_target_stays98, "hour": fact_target_hours98,
        "event": sensor98, "position": np.arange(len(fact_indices98)),
        "state_index": fact_indices98,
        "local_mean": fact_mean98[:, j98]
    }).merge(factual95, on=["stay_id", "hour", "event"], how="inner")
    b98 = 1 + np.searchsorted(
        num_spec0[(sensor98, "value")]["fine_bin_edges"],
        t98.observed.to_numpy(), side="left")
    t98["sensor_index"] = j98
    t98["true_bin"] = b98
    t98["local_nll"] = -np.log(np.maximum(
        fact_prob98[t98.position.to_numpy(), j98, b98], 1e-30))
    t98["mean_error"] = t98.local_mean - t98.predicted_mean
    t98["nll_error"] = t98.local_nll - t98.nll
    fact_parts98.append(t98)
fact_table98 = pd.concat(fact_parts98, ignore_index=True)
print("\nTRUE PRE-READING STATE COMPARISON; seconds:", round(time.time()-t98_start, 1))
print("readings:", len(fact_table98), "external total:", len(factual95))
print(fact_table98.groupby("event").agg(
    local_nll=("local_nll", "mean"), external_nll=("nll", "mean"),
    mean_bias=("mean_error", "mean"),
    mean_RMS=("mean_error", lambda x: np.sqrt(np.mean(x*x))),
    nll_error_RMS=("nll_error", lambda x: np.sqrt(np.mean(x*x)))
).round(5).to_string())
print("OVERALL NLL:", fact_table98.local_nll.mean(),
      "external:", fact_table98.nll.mean())
print("Edited tensors:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 98: rebuilt factual assay\n"
            "- Original core matches saved states. Prior factual assays combined "
            "misordered reading labels and forecasts before intervening actions.\n"
            "- Rebuilt factual targets from consecutive bundles, actual saved states, "
            "and CSV observations. Old candidate NLL diagnostics are not trusted.\n"
            "- No edits. Python runs: 98; forward passes: 4.\n")
```