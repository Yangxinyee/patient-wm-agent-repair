The validated weights are restored exactly. I’ll now try the lower-disturbance norepinephrine decoder at **1.5× the validated dose sensitivity**, rather than 2×. Factual NLL must pass before I test whether its rollout moves closer to the record-derived effect estimate.

```python
import numpy as np

B0_175 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
g175 = np.asarray(candidate_gains173, dtype=np.float64)
entry175 = np.asarray(entry173, dtype=np.float64)

reconstructed173 = B0_175 + entry175 @ np.diag(g175) @ candidate_W173.T
print("Rejected candidate reconstruction error:",
      float(np.max(np.abs(reconstructed173 - candidate_B173))))
print("Insulin decoder difference:",
      float(np.max(np.abs(candidate_W173[:, 0] - validated_W151[:, 0]))))
print("Gains:", g175, "validated:", validated_gains151)
assert np.max(np.abs(reconstructed173 - candidate_B173)) < 2e-5
assert np.max(np.abs(candidate_W173[:, 0] - validated_W151[:, 0])) < 1e-8
assert np.allclose(g175, validated_gains151)

trial_W175 = np.asarray(candidate_W173, dtype=np.float64).copy()
trial_W175[:, 1] *= 0.75
trial_B175 = (B0_175 + entry175 @ np.diag(g175) @ trial_W175.T).astype(np.float32)
save_edit("core.b_in.weight", trial_B175)

print("Installed NE sensitivity: 1.5 times validated doubled-dose probe sensitivity")
print("Weight change from validated, Frobenius norm:",
      float(np.linalg.norm(trial_B175 - validated_B151)))
print("Edited tensors:", edited())
print("Next factual cohort:", list(cohort173))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 175: reduced minimum-disturbance NE trial\n"
            "Reconstructed candidate173 exactly within float32 tolerance. "
            "Insulin decoder and gains are unchanged. Installed the new "
            "NE decoder at 0.75 of candidate173, i.e. 1.5 times validated "
            "doubled-dose probe sensitivity, with gains [-10,+12]. "
            "Require factual budget before rollout. validated_B151 remains "
            "the exact fallback.\n")
```