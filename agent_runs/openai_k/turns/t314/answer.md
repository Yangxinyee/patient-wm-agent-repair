The lightly regularized fits overfit badly. Stronger slot penalties generalized better and improved the dose-response shape, though the effects remain too weak. I’ll verify the best candidates with exact local probabilities, then submit only a bounded-cost trial. The official factual and rollout tests—not these local estimates—will decide whether to retain it.

```python
import numpy as np, torch

r277_candidates = {}
for key in [(0.001,5.0), (0.01,0.5), (0.01,5.0)]:
    r277_candidates[("heldout_fit",)+key] = r276_trials[key]["v"].copy()

# Refit the regularized statistical feature models using all local cases.
r277_Aparts, r277_yparts = [], []
for key in r245_keys:
    A = r276_features[key].mean(axis=0)
    base = r276_base_delta[key].mean(axis=0)
    if key[0] == "norepinephrine":
        desired = r276_coefficients @ r276_rates[key].mean(axis=0)
        y, importance = desired-base, 1.0
    else:
        y, importance = np.array([0.,0.,-base[2],-base[3]]), 0.5
    r277_Aparts.append(np.sqrt(importance)*A)
    r277_yparts.append(np.sqrt(importance)*y)
r277_Aparts.append(2.0*r276_Jf.mean(axis=0))
r277_yparts.append(np.zeros(4))
r277_A = np.concatenate(r277_Aparts)
r277_y = np.concatenate(r277_yparts)

for penalty, softness in [(0.001,5.0),(0.01,0.5),(0.01,5.0),(0.1,5.0)]:
    M = r275_F + penalty*r276_cov + r276_ridge*np.eye(192)
    B = np.linalg.solve(M, r277_A.T)
    v = B @ np.linalg.solve(
        r277_A @ B + softness*np.eye(len(r277_A)), r277_y
    )
    cost = max(0.5*float(v @ r276_Ffit @ v),
               0.5*float(v @ r276_Fval @ v))
    v *= min(1.0, np.sqrt(0.0012/max(cost,1e-15)))
    r277_candidates[("full_fit",penalty,softness)] = v

r277_Ibase = {
    mult: r256_predict(r272_cases[("insulin",mult)], r272_glucose)
    for mult in [2.0,0.5]
}
r277_results = {}
for label, v in r277_candidates.items():
    C = np.outer(r275_direction/r272_delta, v)
    means, probs, _ = r267_full_pre(
        r274_pre(r274_case, r272_glucose+C), indices151
    )
    means = np242(means)
    P = np.stack([np242(p) for p in probs], axis=1)
    kl = np.sum(
        r274_base_P*(np.log(np.maximum(r274_base_P,1e-12)) -
                     np.log(np.maximum(P,1e-12))), axis=2
    )
    report = {
        "weighted_KL": float(kl.mean(axis=0) @ r274_weights),
        "validation_half_KL": float(
            kl[r276_valmask].mean(axis=0) @ r274_weights),
        "factual_RMS": np.sqrt(
            np.mean((means-r274_base_mean)**2,axis=0)
        ).round(5).tolist(),
        "matrix_norm": float(np.linalg.norm(C)),
        "feature_norm": float(np.linalg.norm(v))
    }
    dose_error = 0.0
    for mult, importance in [(0.0,1),(0.5,2),(1.5,1),(2.0,3),(3.0,1)]:
        key = ("norepinephrine",mult)
        dose_means = r256_predict(r272_cases[key], r272_glucose+C)
        shift = (dose_means-means).mean(axis=0)
        target = r276_coefficients @ r276_rates[key].mean(axis=0)
        dose_error += importance*float(
            np.mean((shift[2:]-target[2:])**2)
        )
        report[f"NE_{mult}_exact_shift"] = shift.round(5).tolist()
    cross = []
    for mult in [2.0,0.5]:
        dose_means = r256_predict(
            r272_cases[("insulin",mult)], r272_glucose+C
        )
        added = (
            (dose_means-means) -
            (r277_Ibase[mult]-r274_base_mean)
        ).mean(axis=0)
        cross.append(float(np.max(np.abs(added[:2]))))
        report[f"insulin_{mult}_added_shift"] = added.round(5).tolist()
    report["dose_error"] = dose_error
    report["eligible_for_official_trial"] = bool(
        report["weighted_KL"] <= 0.0016
        and report["validation_half_KL"] <= 0.0022
        and report["feature_norm"] <= 700
        and max(cross) <= 0.3
        and np.isfinite(C).all()
    )
    r277_results[label] = {"correction":C,"v":v,"summary":report}
    print(label, report)

r277_eligible = [
    (label,result) for label,result in r277_results.items()
    if result["summary"]["eligible_for_official_trial"]
]
r277_selected = None
if r277_eligible:
    r277_selected, r277_selected_result = min(
        r277_eligible, key=lambda item: item[1]["summary"]["dose_error"]
    )
    r277_base_weights = np242(load("core.b_in.weight")).copy()
    np.savez(
        "/work/map_feature_trial_backup.npz",
        core_b_in=r277_base_weights
    )
    r277_selected_C = r277_selected_result["correction"]
    r277_trial_weights = (
        r277_base_weights + r277_selected_C
    ).astype(np.float32)
    np.savez(
        "/work/map_feature_trial.npz",
        core_b_in=r277_trial_weights,
        map_correction=r277_selected_C,
        feature=r277_selected_result["v"],
        direction=r275_direction
    )
    save_edit("core.b_in.weight", torch.as_tensor(r277_trial_weights))
    print("SUBMITTED FOR VALIDATION:", r277_selected,
          r277_selected_result["summary"])
else:
    print("No candidate met the local safety bounds; submitted weights unchanged.")

with open("/work/notes.md","a") as f:
    f.write(
        "\n## Run 277: exact local tests and bounded MAP trial\n"
        "- Low-penalty feature fits overfit the local fitting half; rejected those routes.\n"
        "- Exactly evaluated regularized feature candidates and full-local-sample refits across all five NE multipliers and insulin cross-talk.\n"
        "- Any submitted trial has a recoverable glucose-only backup in map_feature_trial_backup.npz.\n"
        "- Selected trial: "+str(r277_selected)+". Official eligibility and rollout response remain unvalidated.\n"
        "- Python runs: 277; forward passes: 25.\n"
    )
```