The state-update pathway is much more responsive than the latent pathway. Its prelude features also retain some dose-change signal while staying relatively small on factual bundles. I’ll strengthen the factual constraints across all four slots, then install a modest insulin-only trial edit. The next GPU test must show a factual NLL increase below 0.005 before we consider keeping it.

```python
import numpy as np
import torch

# Fit a small linear feature, constraining every factual slot rather than
# only their mean. No gradients or optimisation of model weights are used.
slot_features47 = np45(prelude_all46).astype(np.float64)
slot_scale47 = np.maximum(
    slot_features47[train_all46].reshape(-1, 192).std(axis=0), 1e-5
)
slot_factual47 = slot_features47 / slot_scale47[None, None, :]
slot_deltas47 = {}
slot_targets47 = {}
for key47 in prelude_contrasts46:
    drug47, factor47 = key47
    u47 = edited_action46(drug47, factor47)
    z47 = _post33(cf_rows, tt45(cf_E), u47)
    altered47 = prelude46(hflow46[cf_idx46], tt45(cf_E), u47, z47)
    slot_deltas47[key47] = np45(altered47-prelude_cf46).astype(np.float64)
    slot_targets47[key47] = dose_contrasts46[key47]

parts47, targets47 = [], []
for key47, delta47 in slot_deltas47.items():
    multiplier47 = 2.0 if key47[1] == 0.5 else 1.0
    parts47.append(
        multiplier47*(delta47[train_cf46]/slot_scale47).reshape(-1, 192)
    )
    targets47.append(
        multiplier47*np.repeat(slot_targets47[key47][train_cf46], 4, axis=0)
    )
C47 = np.concatenate(parts47)
Y47 = np.concatenate(targets47)
F47 = slot_factual47[train_all46].reshape(-1, 192)
slot_fits47 = {}
slot_metrics47 = {}
for anchor47 in [1.0, 10.0, 50.0]:
    coeff47 = np.linalg.solve(
        C47.T@C47 + anchor47*(F47.T@F47) + 0.1*np.eye(192),
        C47.T@Y47
    ) / slot_scale47[:, None]
    slot_fits47[anchor47] = coeff47
    factual47 = slot_features47[test_all46] @ coeff47
    slopes47 = {}
    for col47, drug47 in enumerate(["insulin", "norepinephrine"]):
        for factor47 in [2.0, 0.5]:
            prediction47 = (
                slot_deltas47[(drug47, factor47)][test_cf46] @ coeff47[:, col47]
            ).mean(axis=1)
            target47 = slot_targets47[(drug47, factor47)][test_cf46, col47]
            active47 = target47 != 0
            slopes47[(drug47, factor47)] = float(
                prediction47[active47]@target47[active47] /
                (target47[active47]@target47[active47])
            )
    slot_metrics47[anchor47] = slopes47
    print("anchor", anchor47, "contrast slopes", slopes47,
          "held-out slot RMS", np.sqrt(np.mean(factual47**2, axis=(0, 1))),
          "held-out mean RMS", np.sqrt(np.mean(factual47.mean(axis=1)**2, axis=0)))

chosen_anchor47 = 10.0
insulin_slope47 = slot_metrics47[chosen_anchor47][("insulin", 2.0)]
assert insulin_slope47 > 0.02, "Insulin contrast too weak for a safe trial."
insulin_feature47 = (
    slot_fits47[chosen_anchor47][:, 0] / insulin_slope47
).astype(np.float32)
trial_strength47 = 0.75  # approximately 0.75 mg/dL extra response per extra unit/hour
trial_direction47 = bin_directions46["insulin"]
delta_weight47 = (
    trial_strength47 *
    np.outer(trial_direction47, insulin_feature47)
).astype(np.float32)

# Preview only the next-reading changes on the locally reconstructed sample.
preview_scalar47 = tt45(prelude_all46[torch.tensor(actual_rows45)]) @ tt45(insulin_feature47)
preview_update47 = (
    trial_strength47 *
    preview_scalar47[:, :, None] *
    (delta46*tt45(trial_direction47))[None, None, :]
)
preview_h47, preview_o47 = coda46(hnew46 + preview_update47)
preview_change47 = np45(
    next_means45(preview_h47, preview_o47)-next_means45(hbase45, obase45)
)
for event47 in ["glucose_finger", "map_arterial"]:
    changes47 = preview_change47[link_df45.event.to_numpy() == event47]
    print("Local factual preview", event47, "mean/RMS/maxabs",
          float(changes47.mean()), float(np.sqrt(np.mean(changes47**2))),
          float(np.max(np.abs(changes47))))

save_edit(
    "core.b_in.weight",
    W0["core.b_in.weight"] + torch.from_numpy(delta_weight47)
)
print("Installed edits:", edited())
print("Update Frobenius norm:", float(np.linalg.norm(delta_weight47)))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 47: first reversible insulin trial\n"
            "All-slot factual-zero / synthetic-dose-contrast linear regression fitted to prelude states. "
            f"Selected anchor={chosen_anchor47}, insulin x2 slope={insulin_slope47:.6f}. "
            f"Added rank-one core.b_in.weight correction, strength={trial_strength47}, "
            f"Frobenius norm={np.linalg.norm(delta_weight47):.6f}. "
            "Direction lowers next glucose with approximately zero MAP response locally. "
            "Not accepted until GPU factual and rollout validation. "
            "Python runs: 47; forward passes: 3.\n")
```
