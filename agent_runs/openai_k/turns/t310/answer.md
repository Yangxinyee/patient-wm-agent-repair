The check stopped before testing probability distortion: the archived matrix is not MAP-only. Its residual is consistent with the separate glucose correction. I’ll subtract that component and repeat the comparison. An exact reconstruction would confirm the bookkeeping explanation; otherwise I’ll keep the two constructions separate.

```python
import numpy as np, torch

r274_direction = np242(rr_Ndirection).astype(np.float64) / r272_delta
r274_ne_matrix = np242(rr_Ntotal).astype(np.float64) - np242(r255_CI)
r274_v = (r274_direction @ r274_ne_matrix) / np.dot(
    r274_direction, r274_direction
)
r274_error = np.linalg.norm(
    r274_ne_matrix - np.outer(r274_direction, r274_v)
) / np.linalg.norm(r274_ne_matrix)
print("MAP-only factorization error:", float(r274_error))
print("MAP-only matrix norm:", float(np.linalg.norm(r274_ne_matrix)))
if r274_error >= 1e-4:
    print("Using retained rr_Nv directly; archive is not treated as rank one.")
    r274_v = np242(rr_Nv).astype(np.float64)

def r274_pre(case, correction):
    injection = np.einsum(
        "bsi,ji->bsj", np242(case["E"]), correction
    ) * r272_delta[None, None, :]
    return case["pre"] + torch.as_tensor(injection, dtype=torch.float32)

r274_case = r272_cases[("factual", 1.0)]
r274_base_mean, r274_base_probs, _ = r267_full_pre(
    r274_pre(r274_case, r272_glucose), indices151
)
r274_base_mean = np242(r274_base_mean)
r274_base_P = np.stack([np242(p) for p in r274_base_probs], axis=1)
r274_weights = np242(r267_sensor_weights).astype(float).copy()
r274_weights /= r274_weights.sum()
r274_holdout = np.ones(len(indices151), dtype=bool)
r274_holdout[np.asarray(r267_sel, dtype=int)] = False

print("Sensor weights:", r274_weights.tolist())
print("Desired direction:", np242(r267_desired).tolist())
print("Held-out local cases:", int(r274_holdout.sum()))

r274_trials = {}
for ridge, direction in r267_routes.items():
    full_correction = np.outer(np242(direction) / r272_delta, r274_v)
    for strength in [0.5, 1.0]:
        correction = strength * full_correction
        means, probs, _ = r267_full_pre(
            r274_pre(r274_case, r272_glucose + correction), indices151
        )
        means = np242(means)
        P = np.stack([np242(p) for p in probs], axis=1)
        kl = np.sum(
            r274_base_P *
            (np.log(np.maximum(r274_base_P, 1e-12)) -
             np.log(np.maximum(P, 1e-12))), axis=2
        )
        result = {
            "KL_by_sensor": kl.mean(axis=0).round(7).tolist(),
            "weighted_KL": float(kl.mean(axis=0) @ r274_weights),
            "heldout_weighted_KL": float(
                kl[r274_holdout].mean(axis=0) @ r274_weights),
            "factual_RMS": np.sqrt(
                np.mean((means-r274_base_mean)**2, axis=0)
            ).round(5).tolist()
        }
        for multiplier in [2.0, 0.5]:
            key = ("norepinephrine", multiplier)
            dose_means = r256_predict(
                r272_cases[key], r272_glucose + correction
            )
            result[f"NE_{multiplier}_shift"] = (
                dose_means-means
            ).mean(axis=0).round(5).tolist()
        r274_trials[(ridge, strength)] = {
            "correction": correction, "summary": result
        }
        print("ridge/strength", ridge, strength, result)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 273-274: separate archived components and compare KL\n"
        "- Run 273 stopped on a rank-one assertion before edits or probability tests.\n"
        "- Run 274 subtracted the archived glucose component before checking the MAP factorization.\n"
        "- Compared probability KL and NE dose responses on 320 local cases, including 296 outside the direction-fitting subset.\n"
        "- No weights changed. Python runs: 274; forward passes: 24.\n"
    )
```