Insulin’s threshold estimates broadly agree with the residual regressions, and the pre-dose and monitoring placebos show no clear jumps. Norepinephrine has no strong discontinuity, so its threshold ratios are unreliable. Next I’ll estimate persistence from dose innovations and later readings, accounting for later doses; agreement across horizons and sensors would support a rollout target.

```python
from sklearn.base import clone
from threadpoolctl import threadpool_limits
import time

def future17(mat, lag):
    z = np.full((30000, 24), np.nan, dtype=np.float64)
    length = min(24, mat.shape[1] - lag)
    z[:, :length] = mat[:, lag:lag + length]
    return z.reshape(-1)

future_res17 = {}
future_models17 = {}
def crossfit17(key, target):
    residual = np.full(len(target), np.nan)
    started17 = time.time()
    for fold17 in [0, 1]:
        tr17 = np.flatnonzero(np.isfinite(target) & (fold_hist != fold17))
        te17 = np.flatnonzero(np.isfinite(target) & (fold_hist == fold17))
        model17 = clone(dml_models[("insulin", 0)])
        model17.fit(Hhist[tr17], target[tr17])
        residual[te17] = target[te17] - model17.predict(Hhist[te17])
        future_models17[(key, fold17)] = model17
    future_res17[key] = residual
    print("Fitted", key, "n", int(np.isfinite(target).sum()),
          "seconds", round(time.time() - started17, 1))
    return residual

with threadpool_limits(limits=8):
    for drug17 in ["insulin", "norepinephrine"]:
        for lag17 in [1, 2]:
            crossfit17((drug17, lag17), future17(A[drug17], lag17))
    for sensor17 in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        for horizon17 in [2, 3]:
            crossfit17((sensor17, horizon17), future17(P[sensor17], horizon17))

persistence17 = {}
for drug17, beta17, sensors17 in [
    ("insulin", -1.95, ["glucose_finger", "glucose_lab"]),
    ("norepinephrine", 3.10, ["map_arterial", "map_cuff"])
]:
    d17 = dml_dose_resid[drug17]
    for sensor17 in sensors17:
        y2 = future_res17[(sensor17, 2)]
        a1 = future_res17[(drug17, 1)]
        ok2 = np.isfinite(y2) & np.isfinite(a1)
        den2 = np.sum(d17[ok2] ** 2)
        rho17 = np.sum(d17[ok2] * (y2[ok2] - beta17 * a1[ok2])) / (beta17 * den2)
        err2 = y2[ok2] - beta17 * a1[ok2] - beta17 * rho17 * d17[ok2]
        sc2 = np.bincount(groups14[ok2], weights=d17[ok2] * err2, minlength=30000)
        se2 = np.sqrt(sc2 @ sc2) / abs(beta17 * den2)
        y3 = future_res17[(sensor17, 3)]
        a2 = future_res17[(drug17, 2)]
        ok3 = np.isfinite(y3) & np.isfinite(a1) & np.isfinite(a2)
        den3 = np.sum(d17[ok3] ** 2)
        g2 = np.sum(d17[ok3] * (
            y3[ok3] - beta17 * a2[ok3] - beta17 * rho17 * a1[ok3]
        )) / den3
        persistence17[sensor17] = {
            "beta_anchor": beta17, "rho_h2": float(rho17),
            "rho_h2_SE": float(se2), "kernel_lag2": float(g2),
            "lag2_over_beta": float(g2 / beta17),
            "rho_squared": float(rho17 ** 2)
        }
        print(sensor17, persistence17[sensor17])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 17\nIndependent RD confirms finger insulin effect around -2 with pre-dose lab and monitoring placebos near zero. NE has no strong threshold first stage, so its RD ratios are NOT evidence. Estimated persistence using cross-fitted current-dose innovations, future outcomes, and future-dose residuals; models are separate record regressions, not world-model training. Results: " + repr(persistence17) + "\n")
```
