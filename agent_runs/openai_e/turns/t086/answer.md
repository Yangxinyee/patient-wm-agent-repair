The dynamic calculation failed to reproduce the established next-hour effect, so I do not accept its negative estimate. I’ll first check stay-hour alignment and residual units, and correct the rollout edit label. If the aligned next-hour calculation recovers roughly +3, I’ll recompute persistence; if not, I’ll revisit identification before editing.

```python
print("s69 index:", s69.index[:6].tolist())
print("Sampled source rows:", sample_idx69[:6].tolist())
print("s69 stay-hours:\n", s69[["stay_id", "t"]].head(6).to_string(index=False))
print("Panel at s69 index:\n", panel.loc[s69.index[:6], ["stay_id", "t"]].to_string(index=False))
print("Panel at sampled rows:\n", panel.loc[sample_idx69[:6], ["stay_id", "t"]].to_string(index=False))
print("Residual-dose SD:", np.std(resid_d69))
print("Residual consistency RMS:", np.sqrt(np.mean((resid_d69 - (d69 - pred_d69))**2)))

lookup74 = panel.set_index(["stay_id", "t"])
assert lookup74.index.is_unique

def aligned_future74(column, horizon):
    keys = pd.MultiIndex.from_arrays([
        s69["stay_id"].to_numpy(), s69["t"].to_numpy() + horizon
    ], names=["stay_id", "t"])
    return lookup74[column].reindex(keys).to_numpy(dtype=float)

aligned_dose74 = aligned_future74("norepinephrine", 0)
print("Aligned current-dose RMS:", np.sqrt(np.nanmean((aligned_dose74 - d69)**2)))
aligned_next74 = aligned_future74("map_any", 1)
old_next74 = np.asarray(targets69["map_any_next"], dtype=float)
joint74 = np.isfinite(aligned_next74) & np.isfinite(old_next74)
print("Next-hour outcome agreement:", {
    "joint_n": int(joint74.sum()),
    "rms": float(np.sqrt(np.mean((aligned_next74[joint74] - old_next74[joint74])**2))),
    "missingness_agreement": float(np.mean(np.isfinite(aligned_next74) == np.isfinite(old_next74)))
})
print("Audited next-hour moment, all rows:",
      moment_ne73(old_next74, pred_y69["map_any_next"], np.ones(len(s69), dtype=bool))[0])
print("Audited next-hour moment, common cohort:",
      moment_ne73(old_next74, pred_y69["map_any_next"], common73)[0])
print("Aligned next-hour moment, common cohort:",
      moment_ne73(aligned_next74, pred_y69["map_any_next"], common73)[0])
print("Actual rollout edit labels:", roll72["edit"].unique().tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 74: reject unreplicated dynamic calculation and audit alignment\n"
            "Run73 did not reproduce the audited next-hour effect; its beta/persistence are "
            "not accepted. Checked source-row versus stay-hour alignment, dose-residual units "
            "and rollout labels before any further fitting. No weights changed. "
            "Python runs: 74; forwards: 10.\n")
```
