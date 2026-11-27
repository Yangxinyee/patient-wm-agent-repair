The innovation fit has encouraging factual recovery, but the reported contrast slopes may mix raw and standardized features. I’ll verify the scaling against the original fit before editing weights. Reproducing the original slopes will confirm that the comparison is valid.

```python
print("FEATURE SCALING AUDIT")
print("slot_scale min/median/max",
      np.min(slot_scale47), np.median(slot_scale47), np.max(slot_scale47))
print("factual standardized reconstruction RMS",
      np.sqrt(np.mean((np.asarray(slot_features47) / np.asarray(slot_scale47)
                       - np.asarray(slot_factual47))**2)))
print("contrast shapes",
      {k: (np.asarray(v).shape, np.asarray(slot_targets47[k]).shape)
       for k, v in slot_deltas47.items()})

def contrast_metrics52(coeff):
    result = {}
    for key, raw_delta in slot_deltas47.items():
        col = 0 if key[0] == "insulin" else 1
        prediction = (
            (np.asarray(raw_delta)[np.asarray(test_cf46, dtype=bool)]
             / np.asarray(slot_scale47)) @ np.asarray(coeff)
        ).mean(axis=1)[:, col]
        truth = np.asarray(slot_targets47[key])[np.asarray(test_cf46, dtype=bool)]
        if truth.ndim == 2:
            truth = truth[:, col]
        active = np.abs(truth) > 1e-8
        result[key] = float(np.dot(prediction[active], truth[active])
                            / np.dot(truth[active], truth[active]))
    return result

print("ORIGINAL FIT SLOPES", contrast_metrics52(slot_fits47[10.0]))
innovation_metrics52 = {}
for anchor, coeff in innovation_fits51.items():
    innovation_metrics52[anchor] = contrast_metrics52(coeff)
    prediction = (np.asarray(slot_factual47)[np.asarray(test_all46, dtype=bool)]
                  @ coeff).mean(axis=1)
    truth = innovation_factual51[np.asarray(test_all46, dtype=bool)]
    print("INNOVATION FIT", anchor, "slopes", innovation_metrics52[anchor],
          "factual RMSE", np.sqrt(np.mean((prediction - truth)**2, axis=0)),
          "target RMS", np.sqrt(np.mean(truth**2, axis=0)),
          "correlations", [np.corrcoef(prediction[:, j], truth[:, j])[0, 1] for j in range(2)])

trial_outer52 = np.outer(np.asarray(trial_direction47), np.asarray(insulin_feature47))
trial_outer_multiplier52 = float(
    np.sum(np.asarray(trial_b_delta) * trial_outer52) / np.sum(trial_outer52**2))
old_unscaled_feature52 = np.asarray(slot_fits47[10.0])[:, 0] / np.asarray(slot_scale47)
old_feature_multiplier52 = float(
    np.dot(np.asarray(insulin_feature47), old_unscaled_feature52)
    / np.dot(old_unscaled_feature52, old_unscaled_feature52))
print("TRIAL FORMULA multipliers:",
      "weight/outer", trial_outer_multiplier52,
      "feature/(coefficient/scale)", old_feature_multiplier52)
print("trial outer reconstruction relative error",
      np.linalg.norm(np.asarray(trial_b_delta) - trial_outer_multiplier52 * trial_outer52)
      / np.linalg.norm(np.asarray(trial_b_delta)))

with open("/work/notes.md", "a") as recovery_log:
    recovery_log.write(
        "\n## Run 52: feature-unit audit\n"
        "Run 51 contrast diagnostics omitted standardization of raw prelude deltas. "
        "Corrected diagnostics and numerically recovered the trial rank-one formula. "
        "No new weight edits. Python runs: 52; forwards: 5.\n"
    )
```