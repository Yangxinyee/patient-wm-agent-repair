After fixing the labels, the provisional sustained effects are about −15.6/+7.8 for insulin and +3.8/−1.9 for norepinephrine on these rollout rows. Both model responses are substantially smaller. Before increasing them, I’ll independently re-estimate the first six lags using stay-separated outcome regressions and the retained dose innovations. Agreement would support the targets; disagreement would argue against calibrating to them.

```python
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

start136 = time.time()
X136 = HX.to_numpy(dtype=np.float32)
sid136 = pan.index.get_level_values("stay_id").to_numpy()
unique_sid136, cluster136 = np.unique(sid136, return_inverse=True)
fold136 = (sid136.astype(np.int64) % 2).astype(np.int8)
nclusters136 = len(unique_sid136)
lp_new136 = {}
lp_models136 = {}

def fit_lp136(drug, target, horizons):
    residuals = dml_residuals[drug]
    original_idx = np.asarray(residuals["indices"], dtype=np.int64)
    original_rd = np.asarray(residuals["rd"], dtype=float)
    baseline = HX[f"last_{target}"].to_numpy(dtype=float)
    dose_series = pan[drug].fillna(0)
    dose0 = dose_series.to_numpy(dtype=float)
    mismatch = np.max(np.abs(dose0[original_idx] -
                             np.asarray(residuals["dose"], dtype=float)))
    print(drug, "dose-fixture max mismatch", round(float(mismatch), 8))
    assert mismatch < 1e-5, "Panel dose does not match the retained instrument."

    future_doses = [
        dose_series.groupby(level="stay_id").shift(-j).to_numpy(dtype=float)
        for j in range(max(horizons))
    ]
    lp_new136.setdefault(drug, {})
    lp_models136.setdefault(drug, {})

    for horizon in horizons:
        future_y = (
            pan[target].groupby(level="stay_id").shift(-horizon)
            .to_numpy(dtype=float)
        )
        keep = (
            np.isfinite(future_y[original_idx])
            & np.isfinite(baseline[original_idx])
            & np.isfinite(future_doses[horizon - 1][original_idx])
        )
        idx = original_idx[keep]
        rd = original_rd[keep]
        y = future_y[idx] - baseline[idx]
        prediction = np.empty(len(idx), dtype=float)
        models = []

        for test_fold in (0, 1):
            test = fold136[idx] == test_fold
            model = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.08,
                max_iter=90, max_leaf_nodes=24, max_bins=127,
                min_samples_leaf=150, l2_regularization=10,
                early_stopping=False,
                random_state=9100 + horizon + 100 * test_fold
            )
            with threadpool_limits(limits=8):
                model.fit(X136[idx[~test]], y[~test])
                prediction[test] = model.predict(X136[idx[test]])
            models.append(model)

        ry = y - prediction
        denominator = np.bincount(
            cluster136[idx], weights=rd * dose0[idx],
            minlength=nclusters136
        )
        numerator = np.bincount(
            cluster136[idx], weights=rd * ry,
            minlength=nclusters136
        )
        exposure_moments = np.stack([
            np.bincount(
                cluster136[idx], weights=rd * future_doses[j][idx],
                minlength=nclusters136
            )
            for j in range(horizon)
        ], axis=1)
        den = denominator.sum()
        response = numerator.sum() / den
        score = numerator - response * denominator
        se = np.sqrt(np.sum(score ** 2)) / abs(den)

        lp_new136[drug][horizon] = {
            "n": len(idx),
            "response": response,
            "se": se,
            "dose_cov": exposure_moments.sum(axis=0) / den,
            "numerator_cluster": numerator,
            "denominator_cluster": denominator,
            "exposure_cluster": exposure_moments,
            "fold_response": [
                float(np.sum(rd[fold136[idx] == f] *
                             ry[fold136[idx] == f]) /
                      np.sum(rd[fold136[idx] == f] *
                             dose0[idx[fold136[idx] == f]]))
                for f in (0, 1)
            ],
        }
        lp_models136[drug][horizon] = models
        print(drug, "h", horizon, "n", len(idx),
              "closed-loop", round(float(response), 4),
              "SE", round(float(se), 4),
              "folds", np.round(lp_new136[drug][horizon]["fold_response"], 4),
              "elapsed", round(time.time() - start136, 1))

def deconvolve_lp136(entries):
    kernel = []
    for h in sorted(entries):
        assert h == len(kernel) + 1
        row = entries[h]
        correction = sum(
            row["dose_cov"][j] * kernel[h - j - 1]
            for j in range(1, h)
        )
        kernel.append(row["response"] - correction)
    return np.asarray(kernel)

for drug, target in [
    ("insulin", "glucose_finger"),
    ("norepinephrine", "map_arterial")
]:
    fit_lp136(drug, target, range(1, 7))
    new_kernel = deconvolve_lp136(lp_new136[drug])
    print(drug, "new direct kernel:", np.round(new_kernel, 4))
    print(drug, "retained kernel:", np.round(provisional_kernels[drug], 4))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n\n## Run 136\n"
        "Independently re-estimated six longitudinal dose-innovation moments "
        "with new stay-separated outcome nuisance regressions. Retained "
        "cluster moment arrays and direct deconvolution in lp_new136. "
        "This checks nuisance sensitivity; linear, time-invariant direct "
        "effects and innovation exogeneity remain identifying assumptions. "
        "No weights changed.\n"
    )
```