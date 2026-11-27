The slot check refutes the simple averaging explanation: the bundle mean itself moves by about 6.1 units before the coda. The regression found directions that look quiet to the immediate decoder but are not safe through recurrence. I’ll add a direct state-change penalty. A useful candidate must retain dose sensitivity while sharply reducing factual state perturbations.

```python
import time
import numpy as np
from scipy.linalg import solve

start150 = time.time()
residual150 = H144 - Ha144 - 30.0 * Hf144
ridge150 = float(np.median(np.diag(residual150)))
ridge_error150 = float(np.max(np.abs(
    residual150 - ridge150 * np.eye(N144)
)))
print("REGRESSION RIDGE", ridge150, "identity residual", ridge_error150)
assert ridge150 >= 0 and ridge_error150 < 1e-5

recovered150 = solve(
    Ha144 + 10.0 * Hf144 + ridge150 * np.eye(N144),
    rhs144.reshape(-1), assume_a="pos", check_finite=False
).reshape(14, 188)
print("Recovered anchor=10 coefficient maximum error",
      float(np.max(np.abs(recovered150 - fits144[10.0]))))
assert np.max(np.abs(recovered150 - fits144[10.0])) < 1e-4

fit_sources150 = np.unique(src141[np.asarray(fit144, dtype=bool)])
held_sources150 = np.unique(src141[np.asarray(held144, dtype=bool)])
slot_design150 = Fslots149[fit_sources150].reshape(-1, 188)
slot_cov150 = slot_design150.T @ slot_design150 / len(slot_design150)
direction_metric150 = Pdir143.T @ Pdir143 / 192.0
Hstate150 = np.kron(direction_metric150, slot_cov150)

fits150, metrics150 = {}, {}
for gamma150 in (.003, .01, .03, .1, .3, 1.0, 3.0):
    matrix150 = (
        Ha144 + 10.0 * Hf144 + gamma150 * Hstate150
        + ridge150 * np.eye(N144)
    )
    coef150 = solve(
        matrix150, rhs144.reshape(-1),
        assume_a="pos", check_finite=False
    ).reshape(14, 188)
    fits150[gamma150] = coef150

    held_shift150 = (
        Fslots149[held_sources150] @ coef150.T
    ) @ Pdir143.T
    amplitudes_read150 = Fread144 @ coef150.T
    kl150 = .5 * np.einsum(
        "ni,nij,nj->n", amplitudes_read150, fisher143, amplitudes_read150
    )

    report150 = {
        "weight_norm": float(np.linalg.norm(coef150)),
        "heldout_state_rms": float(np.sqrt(np.mean(held_shift150 ** 2))),
        "heldout_local_KL": float(np.mean(kl150[held144])),
    }
    for group150 in (0, 1, 2, 4):
        prediction150 = np.sum(
            (Xparts144[group150] @ coef150.T) * Jparts144[group150], axis=1
        )
        info150 = group_info144[group150]
        beta150 = float(info150[-1])
        if beta150:
            dose_change150 = Yparts144[group150] / beta150
            report150[str(info150[0]) + "_slope"] = float(
                prediction150 @ dose_change150
                / max(dose_change150 @ dose_change150, 1e-12)
            )
        else:
            report150[str(info150[0]) + "_guard_rms"] = float(
                np.sqrt(np.mean(prediction150 ** 2))
            )
    metrics150[gamma150] = report150
    print(gamma150, report150)

print("Elapsed seconds", round(time.time() - start150, 2))
with open("/work/notes.md", "a") as f150:
    f150.write(
        "\n## Run 150: constrain factual state changes\n"
        "Run149 showed pre-coda slot RMS=6.145 and bundle-mean RMS=6.094; "
        "large decoder-insensitive changes, not mainly slot cancellation. "
        "Recovered the prior normal equations and fit a closed-form frontier "
        "with an additional per-slot pre-coda state penalty. "
        f"Frontier: {metrics150}. No weights changed. "
        "Python runs: 150; forwards: 16.\n"
    )
```