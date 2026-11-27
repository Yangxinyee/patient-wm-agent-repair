The recovered arrays include both synthetic dose contrasts and record-based dose innovations. I’ll fit a feature that preserves the contrasts while targeting those innovations on factual bundles. If it cannot recover innovations on held-out stays, I won’t use it as a causal correction.

```python
print("DOSE / RAW DOSE / INNOVATION EXAMPLES")
print(np.column_stack([cf_doses[:8], action_raw[:8], action_targets[:8]]))
print("dose/raw RMS difference",
      np.sqrt(np.mean((np.asarray(cf_doses) - np.asarray(action_raw))**2, axis=0)))
print("innovation SD", np.std(action_targets, axis=0))
print("local response of current direction",
      np.asarray(J46) @ np.asarray(trial_direction47))

innovation_factual51 = np.zeros((len(slot_factual47), 2), dtype=np.float64)
innovation_factual51[np.asarray(cf_rows, dtype=int)] = np.asarray(action_targets)
innovation_F51 = np.asarray(slot_factual47)[np.asarray(train_all46, dtype=bool)].reshape(-1, 192).astype(np.float64)
innovation_R51 = np.repeat(innovation_factual51[np.asarray(train_all46, dtype=bool)], 4, axis=0)
innovation_C51 = np.asarray(C47, dtype=np.float64)
innovation_Y51 = np.asarray(Y47, dtype=np.float64)
innovation_fits51 = {}

for innovation_anchor51 in [1.0, 10.0, 50.0]:
    innovation_W51 = np.linalg.solve(
        innovation_C51.T @ innovation_C51
        + innovation_anchor51 * innovation_F51.T @ innovation_F51
        + 0.1 * np.eye(192),
        innovation_C51.T @ innovation_Y51
        + innovation_anchor51 * innovation_F51.T @ innovation_R51
    )
    innovation_fits51[innovation_anchor51] = innovation_W51
    innovation_test_pred51 = (
        np.asarray(slot_factual47)[np.asarray(test_all46, dtype=bool)] @ innovation_W51
    ).mean(axis=1)
    innovation_test_truth51 = innovation_factual51[np.asarray(test_all46, dtype=bool)]
    innovation_slopes51 = {}
    for innovation_key51, innovation_delta51 in slot_deltas47.items():
        innovation_col51 = 0 if innovation_key51[0] == "insulin" else 1
        innovation_contrast51 = (
            np.asarray(innovation_delta51)[np.asarray(test_cf46, dtype=bool)]
            @ innovation_W51
        ).mean(axis=1)[:, innovation_col51]
        innovation_truth51 = np.asarray(slot_targets47[innovation_key51])[np.asarray(test_cf46, dtype=bool)]
        if innovation_truth51.ndim == 2:
            innovation_truth51 = innovation_truth51[:, innovation_col51]
        innovation_active51 = np.abs(innovation_truth51) > 1e-8
        innovation_slopes51[innovation_key51] = float(
            np.dot(innovation_contrast51[innovation_active51], innovation_truth51[innovation_active51])
            / np.dot(innovation_truth51[innovation_active51], innovation_truth51[innovation_active51])
        )
    print("anchor", innovation_anchor51,
          "factual target RMSE", np.sqrt(np.mean((innovation_test_pred51 - innovation_test_truth51)**2, axis=0)),
          "factual RMS", np.sqrt(np.mean(innovation_test_pred51**2, axis=0)),
          "contrast slopes", innovation_slopes51)

with open("/work/notes.md", "a") as recovery_log:
    recovery_log.write(
        "\n## Run 51: dose-innovation feature comparison\n"
        "Closed-form statistical fits replace factual-zero targets with "
        "record-based action-dose innovations. No additional weight edits. "
        "Python runs: 51; forwards: 5.\n"
    )
```