Norepinephrine’s late-hour arterial estimates are near zero at lags 1, 4 and 8, but lag 12 has a positive signal; cuff readings also show a weak signal. I won’t dismiss these. Next I’ll finish insulin’s delayed checks and examine dose-context differences. Consistent effects across contexts or adjacent lags would challenge the null hypothesis.

```python
import time, json
import numpy as np
from pathlib import Path

analysis_started = time.time()
insulin_delayed_results_v2 = []
for horizon in (8, 12):
    Y = future_reading_v2(0, horizon)
    eligible = positive_insulin_v2 & np.isfinite(Y)
    pA = crossfit_mean_v2(X_insulin_v2, A_insulin_v2, eligible,
                         f"insulin dose horizon {horizon}")
    pY = crossfit_mean_v2(X_insulin_v2, Y, eligible,
                         f"glucose horizon {horizon}")
    rA, rY = A_insulin_v2 - pA, Y - pY
    insulin_y_residuals_v2[horizon] = rY
    for label, selection in {
        "all_positive": eligible,
        "hour_ge4": eligible & (treatment_hours_v2 >= 4),
        "hour_ge12": eligible & (treatment_hours_v2 >= 12)
    }.items():
        result = {"drug": "insulin", "outcome": "glucose_finger",
                  "horizon": horizon, "subset": label,
                  **orthogonal_coefficient_v2(rA, rY, selection)}
        insulin_delayed_results_v2.append(result)
        print(json.dumps(result), flush=True)

context_results_v2 = []
for drug, pA, rA, rY, boundaries in [
    ("insulin", pA_insulin_v2, rA_insulin_v2, insulin_y_residuals_v2[1],
     [0, 1.5, 2.5, 4, np.inf]),
    ("norepinephrine", ne_residuals_v2[(2, 1)]["pa"],
     ne_residuals_v2[(2, 1)]["a"], ne_residuals_v2[(2, 1)]["y"],
     [0, .25, .5, .75, np.inf])
]:
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        selection = (treatment_hours_v2 >= 12) & (pA >= lo) & (pA < hi)
        if np.sum(selection & np.isfinite(rY)) < 500:
            continue
        result = {"drug": drug, "horizon": 1,
                  "subset": f"late_predicted_rate_{lo}_to_{hi}",
                  **orthogonal_coefficient_v2(rA, rY, selection)}
        context_results_v2.append(result)
        print(json.dumps(result), flush=True)

print("\n=== NE POLICY / INNOVATION DIAGNOSTICS ===")
previous_ne = arr[:, 0:24, 5].reshape(-1)
eligible = positive_ne_v2 & (treatment_hours_v2 >= 12)
for label, selection in {
    "all_late_positive": eligible,
    "late_positive_unsaturated": eligible & (A_ne_v2 < .999),
    "late_saturated": eligible & (A_ne_v2 >= .999),
    "late_previous_positive_unsaturated": eligible & (previous_ne > 0)
        & (A_ne_v2 < .999)
}.items():
    innovation = (A_ne_v2 - previous_ne)[selection]
    resid = ne_residuals_v2[(2, 1)]["a"][selection]
    resid = resid[np.isfinite(resid)]
    print(label, "n:", len(innovation),
          "rate-change quantiles:", np.quantile(innovation, [.05,.25,.5,.75,.95]).tolist(),
          "rate-residual mean/sd:", [float(np.mean(resid)), float(np.std(resid))])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Delayed insulin and dose-context checks\n"
            + json.dumps(insulin_delayed_results_v2 + context_results_v2) + "\n"
            "Important NE counter-evidence: late arterial lag-12 estimate "
            "+1.526 (SE 0.656), based only on doses at hour 12; late cuff lag-1 "
            "+0.426 (SE 0.212), weaker when saturated doses are excluded. "
            "These are not proof of an effect or proof of confounding. "
            "Next check adjacent long lags and stronger, late-only adjustment before deciding.\n")
print("Completed in", round(time.time() - analysis_started, 1), "seconds")
```
