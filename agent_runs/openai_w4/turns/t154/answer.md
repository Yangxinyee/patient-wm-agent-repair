The new regressions closely reproduce the first six lags for both drugs, including insulin’s delayed peak. That strengthens the record-based targets, though it does not remove the identification assumptions. I’ll now estimate lags 7–12, quantify stay-level uncertainty, and test immediate dose nonlinearity. A different tail or strong curvature would require revising the rollout targets before changing weights.

```python
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

start136 = time.time()
for drug, target in [
    ("insulin", "glucose_finger"),
    ("norepinephrine", "map_arterial")
]:
    fit_lp136(drug, target, range(7, 13))

rng137 = np.random.default_rng(19371)
boot_weights137 = rng137.poisson(
    1, size=(300, nclusters136)
).astype(np.float32)
kernels137 = {}
boot_kernels137 = {}
readout_estimates137 = {}

print("\n=== TWELVE-LAG DIRECT EFFECTS AND CLUSTER INTERVALS ===")
for drug in ["insulin", "norepinephrine"]:
    entries = lp_new136[drug]
    kernel = deconvolve_lp136(entries)
    boot_kernel = np.zeros((len(boot_weights137), 12))
    for h in range(1, 13):
        row = entries[h]
        den = boot_weights137 @ row["denominator_cluster"]
        response = (boot_weights137 @ row["numerator_cluster"]) / den
        covariances = (
            boot_weights137 @ row["exposure_cluster"]
        ) / den[:, None]
        correction = sum(
            covariances[:, j] * boot_kernel[:, h - j - 1]
            for j in range(1, h)
        )
        boot_kernel[:, h - 1] = response - correction

    kernels137[drug] = kernel
    boot_kernels137[drug] = boot_kernel
    intervals = np.quantile(boot_kernel, [0.025, 0.975], axis=0)
    print(drug)
    print(pd.DataFrame({
        "lag": np.arange(1, 13),
        "direct": kernel,
        "lo": intervals[0],
        "hi": intervals[1],
        "previous_two_pole": np.asarray(kernel_models88[drug])[:12]
    }).round(4).to_string(index=False))

    rates = (
        drug_rows134.loc[drug_rows134.event.eq(drug)]
        .pivot_table(index="stay_id", columns="start_hour",
                     values="rate", aggfunc="sum")
        .reindex(index=stay_ids134, columns=range(26))
        .fillna(0).to_numpy()
    )
    edit = "insulin_x2" if drug == "insulin" else "ne_x2"
    rows = active135.loc[active135.edit.eq(edit)].copy()
    row_stays = np.array([stay_lookup134[sid] for sid in rows.stay_id])
    row_hours = np.rint(rows.hour).astype(int).to_numpy()
    exposure = np.zeros((len(rows), 12))
    for lag in range(1, 13):
        starts = row_hours - lag
        valid = starts >= 12
        exposure[valid, lag - 1] = rates[row_stays[valid], starts[valid]]

    row_average = exposure.mean(axis=0)
    exposure_frame = pd.DataFrame(exposure)
    exposure_frame["stay_id"] = rows.stay_id.to_numpy()
    stay_average = exposure_frame.groupby("stay_id").mean().mean().to_numpy()

    estimates = {}
    for label, weights in [("row_mean", row_average),
                           ("equal_stay_mean", stay_average)]:
        estimate = float(weights @ kernel)
        samples = boot_kernel @ weights
        interval = np.quantile(samples, [0.025, 0.975])
        estimates[label] = (estimate, interval)
        print(label, "x2", round(estimate, 4),
              "95% cluster interval", np.round(interval, 4),
              "half", round(-0.5 * estimate, 4))
    readout_estimates137[drug] = estimates

print("\n=== IMMEDIATE LINEAR-VERSUS-QUADRATIC DOSE TEST ===")
quadratic137 = {}
for drug in ["insulin", "norepinephrine"]:
    r = dml_residuals[drug]
    idx = np.asarray(r["indices"], dtype=np.int64)
    dose = np.asarray(r["dose"], dtype=float)
    rd = np.asarray(r["rd"], dtype=float)
    ry = np.asarray(r["ry"], dtype=float)
    square_prediction = np.empty(len(idx))

    for test_fold in (0, 1):
        test = fold136[idx] == test_fold
        model = HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=120,
            max_leaf_nodes=24, max_bins=127,
            min_samples_leaf=150, l2_regularization=10,
            early_stopping=False, random_state=19800 + test_fold
        )
        with threadpool_limits(limits=8):
            model.fit(X136[idx[~test]], dose[~test] ** 2)
            square_prediction[test] = model.predict(X136[idx[test]])

    treatment_residuals = np.column_stack([
        rd, dose ** 2 - square_prediction
    ])
    gram_inverse = np.linalg.pinv(treatment_residuals.T @ treatment_residuals)
    beta = gram_inverse @ treatment_residuals.T @ ry
    error = ry - treatment_residuals @ beta
    scores = np.column_stack([
        np.bincount(cluster136[idx],
                    weights=treatment_residuals[:, j] * error,
                    minlength=nclusters136)
        for j in range(2)
    ])
    covariance = gram_inverse @ (scores.T @ scores) @ gram_inverse
    se = np.sqrt(np.diag(covariance))
    quadratic137[drug] = {"beta": beta, "se": se}
    print(drug, "linear/quadratic coefficients", np.round(beta, 5),
          "cluster SE", np.round(se, 5),
          "positive-dose percentiles",
          np.round(np.quantile(dose[dose > 0], [0.05, 0.5, 0.95, 1]), 4))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 137\n"
        "Extended independently fitted innovation local projections through "
        "12 lags; computed 300 stay-cluster multiplier-bootstrap direct-kernel "
        "and matched-readout intervals. Tested a quadratic immediate dose "
        "term with cross-fitted dose-squared nuisance estimates. These "
        "estimates still depend on sequential exogeneity, sufficient history "
        "and an additive transferable impulse response. No weights changed.\n"
    )
```