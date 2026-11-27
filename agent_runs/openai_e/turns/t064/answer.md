The unit audit rules out projection alone: it greatly increases leakage on later bundles. The constrained refit is more promising, with zero admission response and less held-out leakage per unit of insulin sensitivity. I’ll install a reversible stronger version, aiming to bring insulin x2 near the provisional record-based target while preserving factual fit.

```python
import numpy as np
import torch

candidate_kind58 = ("zero", 10.0)
candidate_raw58 = constrained_fits57[candidate_kind58][:, 0].copy()
direction58 = np.asarray(trial_direction47, dtype=np.float64)
old_raw58 = (
    direction58 @ delta56 /
    (trial_strength47 * (direction58 @ direction58))
)

def full_insulin_slope58(coefficient):
    delta58 = np.asarray(slot_deltas47[("insulin", 2.0)])
    prediction58 = (delta58 @ coefficient).mean(axis=1)
    dose58 = cf_doses[:, 0].astype(np.float64)
    active58 = dose58 > 0
    return float(prediction58[active58] @ dose58[active58] /
                 (dose58[active58] @ dose58[active58]))

old_slope58 = full_insulin_slope58(old_raw58)
new_slope58 = full_insulin_slope58(candidate_raw58)
normalization58 = old_slope58 / new_slope58
strength_multiplier58 = 2.0
candidate_input58 = normalization58 * candidate_raw58
candidate_delta58 = (
    trial_strength47 * strength_multiplier58 *
    np.outer(direction58, candidate_input58)
).astype(np.float32)
candidate_b58 = (
    load_original("core.b_in.weight") +
    torch.from_numpy(candidate_delta58)
)
save_edit("core.b_in.weight", candidate_b58)

print("INSTALLED CONSTRAINED INSULIN CANDIDATE")
print("fit:", candidate_kind58)
print("old/new unnormalized x2 slopes:", old_slope58, new_slope58)
print("feature normalization:", normalization58)
print("strength relative to old trial:", strength_multiplier58)
print("delta Frobenius norm:", float(np.linalg.norm(candidate_delta58)))
print("old physical feature:", feature_metrics57(old_raw58))
print("new normalized feature:", feature_metrics57(candidate_input58))
print("initial feature RMS:",
      float(np.sqrt(np.mean((initial_matrix56 @ candidate_input58)**2))))
print("edited tensors:", edited())

np.savez(
    "/work/insulin_candidate58.npz",
    delta=candidate_delta58,
    input_feature=candidate_input58,
    direction=direction58,
    normalization=np.array(normalization58),
    strength_multiplier=np.array(strength_multiplier58),
    initial_basis=initial_basis56,
    original_trial_delta=delta56.astype(np.float32),
)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 58: installed admission-null insulin candidate\n"
        "Projection alone failed the unit-correct factual audit. Selected "
        "constrained factual-zero fit anchor=10, with exact initial-prelude "
        "null constraints. Normalized its insulin-x2 contrast to the old "
        f"trial (factor {normalization58:.8g}), then doubled trial strength. "
        "Edit formula: original b_in + .75*2*outer(trial_direction47, "
        "normalization58*constrained_fits57[('zero',10)][:,0]). "
        f"Delta norm={np.linalg.norm(candidate_delta58):.6g}. "
        "Saved /work/insulin_candidate58.npz. Candidate is NOT accepted "
        "until GPU factual and paired rollout validation. Provisional "
        "training-record insulin targets: x2 about -13.6, half about +6.8. "
        "NE identification/persistence remains to audit. "
        "Python runs: 58; forwards: 5.\n"
    )
```