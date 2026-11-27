The audit stopped because some reading rows have no entry in the filtered state table. No prediction comparison or weight change occurred. I’ll rebuild the index map, explicitly exclude missing entries, and finish the same-reading comparison. If that resolves the NLL discrepancy, the previous preservation scores were indexing artifacts.

```python
import numpy as np, pandas as pd, torch, dis

inverse97 = np.full(len(rr33_stays), -1, dtype=int)
inverse97[np.asarray(idx80, dtype=int)] = np.arange(len(idx80))
mapped97 = inverse97[np.asarray(rows80, dtype=int)]
valid97 = mapped97 >= 0
print("MAPPING AUDIT")
print("missing rows:", int((~valid97).sum()),
      "old/new mapping disagreements:", int(np.sum(inverse97 != lookup80)))
print("missing stay/hour:", list(zip(
    rr33_stays[rows80[~valid97]], rr33_hours[rows80[~valid97]]))[:12])
print("idx80 endpoints:", idx80[:8], idx80[-8:])
print("future target count:", len(future_index59), "state row count:", len(rows80))

audit_parts97 = []
target_stays97 = np.asarray(future_index59.get_level_values(0))[valid97]
target_hours97 = np.asarray(future_index59.get_level_values(1))[valid97]
for j97, sensor97 in enumerate(r49_sensors):
    t97 = pd.DataFrame({
        "stay_id": target_stays97,
        "hour": target_hours97,
        "event": sensor97,
        "local_mean": M0_80[mapped97[valid97], j97],
        "local_position": mapped97[valid97]
    }).merge(factual95, on=["stay_id", "hour", "event"], how="inner")
    b97 = 1 + np.searchsorted(
        num_spec0[(sensor97, "value")]["fine_bin_edges"],
        t97.observed.to_numpy(), side="left")
    p97 = probs0_80[t97.local_position.to_numpy(), j97, b97]
    t97["local_nll"] = -np.log(np.maximum(p97, 1e-30))
    t97["bin_from_record"] = b97
    t97["mean_error"] = t97.local_mean - t97.predicted_mean
    t97["nll_error"] = t97.local_nll - t97.nll
    audit_parts97.append(t97)
audit97 = pd.concat(audit_parts97, ignore_index=True)
print("\nEXACT-KEY COMPARISON")
print("readings:", len(audit97))
print(audit97.groupby("event").agg(
    local_nll=("local_nll", "mean"),
    external_nll=("nll", "mean"),
    mean_bias=("mean_error", "mean"),
    mean_RMS=("mean_error", lambda x: np.sqrt(np.mean(x*x))),
    nll_error_RMS=("nll_error", lambda x: np.sqrt(np.mean(x*x)))
).round(5).to_string())
print(audit97.head(8)[["stay_id", "hour", "event", "observed",
                       "bin_from_record", "local_mean", "predicted_mean",
                       "local_nll", "nll"]].to_string(index=False))

print("\nFRESH ORIGINAL READOUT CHECK")
ix97 = rows80[valid97][:24]
out97, h97 = r59_core(ix97)
features97 = r59_features(out97, h97, ix97)
print("returned shapes:", [tuple(v.shape) for v in features97])
fresh_mean97 = np.asarray(features97[0])
print("fresh/saved mean max difference:",
      np.max(np.abs(fresh_mean97 - M0_80[inverse97[ix97]])))
print("fresh/saved state max differences:",
      float(torch.max(torch.abs(out97 - torch.tensor(sx[ix97, :192])))),
      float(torch.max(torch.abs(h97.reshape(-1, 768) -
                                torch.tensor(sx[ix97, 192:960])))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 96–97: filtered-state mapping audit\n"
            "- Run 96 stopped at an assertion: some reading rows were missing from "
            "the filtered state table. No predictions or weights changed.\n"
            "- Rebuilt mapping directly from idx80, excluded missing entries, and "
            "completed exact-key comparison plus fresh original reconstruction.\n"
            "- No edits. Python runs: 97; forward passes: 4.\n")
```