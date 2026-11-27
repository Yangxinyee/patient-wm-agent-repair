The new routes strengthen both norepinephrine responses, but mean shifts alone cannot show whether they preserve factual fit. The reconstruction also differs from the archived MAP correction by 20%. I’ll resolve that difference and compare probability distortion on the larger local sample; a genuinely lower-cost route should retain its dose response there.

```python
import numpy as np, torch

r273_old_direction = np242(rr_Ndirection).astype(np.float64) / r272_delta
r273_total = np242(rr_Ntotal).astype(np.float64)
r273_v = (r273_old_direction @ r273_total) / np.dot(
    r273_old_direction, r273_old_direction
)
r273_factor_error = np.linalg.norm(
    r273_total - np.outer(r273_old_direction, r273_v)
) / np.linalg.norm(r273_total)
print("Archived MAP rank-one reconstruction error:", r273_factor_error)
print("Recovered feature versus rr_Nv:",
      {"relative_difference": float(np.linalg.norm(r273_v-r272_v) /
                                    np.linalg.norm(r272_v)),
       "norm": float(np.linalg.norm(r273_v))})
assert r273_factor_error < 1e-4, "Archived MAP correction is not rank one."

def r273_pre(case, correction):
    injection = np.einsum(
        "bsi,ji->bsj", np242(case["E"]), correction
    ) * r272_delta[None, None, :]
    return case["pre"] + torch.as_tensor(injection, dtype=torch.float32)

r273_factual_case = r272_cases[("factual", 1.0)]
r273_base_mean, r273_base_probs, _ = r267_full_pre(
    r273_pre(r273_factual_case, r272_glucose), indices151
)
r273_base_P = np.stack([np242(p) for p in r273_base_probs], axis=1)
r273_base_mean = np242(r273_base_mean)
r273_weights = np242(r267_sensor_weights).astype(float)
r273_weights /= r273_weights.sum()
r273_holdout = np.ones(len(indices151), dtype=bool)
r273_holdout[np.asarray(r267_sel, dtype=int)] = False

print("Sensor weights:", r273_weights.tolist())
print("Desired readout direction:", np242(r267_desired).tolist())
print("Held-out local cases:", int(r273_holdout.sum()))

r273_trials = {}
for ridge, direction in r267_routes.items():
    full_correction = np.outer(np242(direction) / r272_delta, r273_v)
    for strength in [0.5, 1.0]:
        correction = strength * full_correction
        means, probs, _ = r267_full_pre(
            r273_pre(r273_factual_case, r272_glucose + correction), indices151
        )
        P = np.stack([np242(p) for p in probs], axis=1)
        means = np242(means)
        kl = np.sum(
            r273_base_P *
            (np.log(np.maximum(r273_base_P, 1e-12)) -
             np.log(np.maximum(P, 1e-12))), axis=2
        )
        result = {
            "KL_by_sensor": kl.mean(axis=0).round(7).tolist(),
            "weighted_KL": float(kl.mean(axis=0) @ r273_weights),
            "heldout_weighted_KL": float(
                kl[r273_holdout].mean(axis=0) @ r273_weights),
            "factual_RMS": np.sqrt(
                np.mean((means-r273_base_mean)**2, axis=0)
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
        r273_trials[(ridge, strength)] = {
            "correction": correction, "summary": result
        }
        print("ridge/strength", ridge, strength, result)

print("Current submitted tensors:", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 273: centered feature and probability-distortion check\n"
        "- Reconstructed the archived MAP correction's effective feature algebraically.\n"
        "- Compared categorical KL and dose responses for Fisher routes on 320 local cases, including cases outside the 24 used to fit directions.\n"
        "- KL is a diagnostic, not the official factual value NLL or all-output log loss.\n"
        "- No weights changed. Python runs: 273; forward passes: 24.\n"
    )
```