The admission constraints remove the initial feature response without losing the insulin contrast. Projection alone may still introduce errors elsewhere, so I’ll refit the feature under those constraints and check it on held-out stays. I’ll also audit the units: the useful comparison is factual leakage per unit of dose sensitivity.

```python
import numpy as np

raw_E57 = prelude_all46.detach().cpu().numpy().astype(np.float64)
scale57 = np.asarray(slot_scale47, dtype=np.float64)
print("UNIT AUDIT")
print("slot_factual versus raw/scale RMS:",
      float(np.sqrt(np.mean((slot_factual47 - raw_E57 / scale57)**2))))
print("slot_features versus raw RMS:",
      float(np.sqrt(np.mean((slot_features47 - raw_E57)**2))))

drug_rows57 = np.flatnonzero(
    state_df.has_insulin.to_numpy() | state_df.has_norepinephrine.to_numpy()
)
assert len(drug_rows57) == len(cf_doses)
print("dose/flag agreement:",
      float(np.mean((cf_doses[:, 0] > 0) ==
                    state_df.has_insulin.to_numpy()[drug_rows57])),
      float(np.mean((cf_doses[:, 1] > 0) ==
                    state_df.has_norepinephrine.to_numpy()[drug_rows57])))

fit_stays57 = np.sort(state_df.stay_id.unique())[:80]
fit_rows57 = state_df.stay_id.isin(fit_stays57).to_numpy()
fit_drug57 = fit_rows57[drug_rows57]
F57 = (raw_E57[fit_rows57] / scale57).reshape(-1, 192)
Ftest57 = raw_E57[~fit_rows57]
Cparts57, Yparts57 = [], []
targets57 = {}
for key57, delta57 in slot_deltas47.items():
    drug57, factor57 = key57
    target57 = np.zeros((len(cf_doses), 2), dtype=np.float64)
    column57 = 0 if drug57 == "insulin" else 1
    target57[:, column57] = (factor57 - 1.0) * cf_doses[:, column57]
    targets57[key57] = target57
    Cparts57.append((np.asarray(delta57)[fit_drug57] / scale57).reshape(-1, 192))
    Yparts57.append(np.repeat(target57[fit_drug57], 4, axis=0))
C57 = np.concatenate(Cparts57)
Y57 = np.concatenate(Yparts57)
Gc57 = C57.T @ C57
Gf57 = F57.T @ F57
Rc57 = C57.T @ Y57

initial_std57 = initial_matrix56 / scale57
_, singular_init57, vt_init57 = np.linalg.svd(initial_std57, full_matrices=False)
rank_init57 = int(np.sum(singular_init57 > singular_init57[0] * 1e-6))
null57 = vt_init57[rank_init57:].T

def feature_metrics57(raw_coefficient):
    factual57 = Ftest57 @ raw_coefficient
    result57 = {
        "heldout_slot_rms": float(np.sqrt(np.mean(factual57**2))),
        "heldout_mean_rms": float(np.sqrt(np.mean(factual57.mean(axis=1)**2))),
        "initial_rms": float(np.sqrt(np.mean((initial_matrix56 @ raw_coefficient)**2))),
        "norm": float(np.linalg.norm(raw_coefficient)),
    }
    for key57, delta57 in slot_deltas47.items():
        column57 = 0 if key57[0] == "insulin" else 1
        dose_delta57 = targets57[key57][:, column57]
        active57 = (~fit_drug57) & (np.abs(dose_delta57) > 0)
        predicted57 = (np.asarray(delta57) @ raw_coefficient).mean(axis=1)
        result57[str(key57)] = float(
            predicted57[active57] @ dose_delta57[active57] /
            max(dose_delta57[active57] @ dose_delta57[active57], 1e-12)
        )
    return result57

print("\nORIGINAL AND PROJECTED TRIAL INPUT FEATURES")
print("original:", feature_metrics57(trial_input56))
print("projected:", feature_metrics57(projected_input56))

constrained_fits57 = {}
constrained_metrics57 = {}
for kind57 in ("zero", "innovation"):
    if kind57 == "zero":
        Rf57 = np.zeros((192, 2))
    else:
        factual_targets57 = np.repeat(
            np.asarray(innovation_factual51)[fit_rows57], 4, axis=0
        )
        Rf57 = F57.T @ factual_targets57
    for anchor57 in (1.0, 10.0, 50.0):
        G57 = Gc57 + anchor57 * Gf57 + 0.1 * np.eye(192)
        R57 = Rc57 + anchor57 * Rf57
        coefficients57 = null57 @ np.linalg.solve(
            null57.T @ G57 @ null57, null57.T @ R57
        )
        raw_coefficients57 = coefficients57 / scale57[:, None]
        constrained_fits57[(kind57, anchor57)] = raw_coefficients57
        metrics57 = feature_metrics57(raw_coefficients57[:, 0])
        constrained_metrics57[(kind57, anchor57)] = metrics57
        print(kind57, anchor57, metrics57)

print("No weights changed.")
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 57: constrained, unit-audited feature refits\n"
        "Initial prelude has rank 4. Euclidean projection retained drug "
        "contrasts but was not accepted without a consistent factual-unit "
        "audit. Fitted closed-form two-output regressions with exact "
        "admission-null constraints, 80 fitting stays and 20 held-out stays; "
        "compared factual-zero and record-based innovation targets. "
        "Metrics: " + repr(constrained_metrics57) + "\n"
        "No weights changed. Python runs: 57; forwards: 5.\n"
    )
```