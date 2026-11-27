Insulin’s late-hour estimates are near zero at both one hour and four hours, including the current-reading and lower-rate checks. This supports the edit, though it does not prove an exact null. I’ll test norepinephrine with the same pre-dose design, including longer horizons; a persistent late-hour MAP effect would argue for retaining dose sensitivity.

```python
import time, json
import numpy as np
from pathlib import Path

analysis_started = time.time()
X_ne_v2, ne_features_v2 = pre_dose_features_v2("norepinephrine")
A_ne_v2 = arr[:, 1:25, 5].reshape(-1).astype(np.float64)
positive_ne_v2 = np.isfinite(A_ne_v2) & (A_ne_v2 > 0)
ne_results_v2 = []
ne_residuals_v2 = {}

for channel, horizon in [(2, 1), (2, 4), (2, 8), (2, 12), (3, 1)]:
    Y = future_reading_v2(channel, horizon)
    eligible = positive_ne_v2 & np.isfinite(Y)
    # Match both nuisance regressions to the same observed-outcome population.
    pA = crossfit_mean_v2(X_ne_v2, A_ne_v2, eligible,
                         f"NE dose {events[channel]} horizon {horizon}")
    pY = crossfit_mean_v2(X_ne_v2, Y, eligible,
                         f"{events[channel]} horizon {horizon}")
    rA, rY = A_ne_v2 - pA, Y - pY
    ne_residuals_v2[(channel, horizon)] = {"a": rA, "y": rY, "pa": pA}
    selections = {
        "all_positive": eligible,
        "hour_lt4": eligible & (treatment_hours_v2 < 4),
        "hour_ge4": eligible & (treatment_hours_v2 >= 4),
        "hour_ge12": eligible & (treatment_hours_v2 >= 12),
        "hour_ge12_unsaturated": eligible & (treatment_hours_v2 >= 12)
            & (A_ne_v2 < 0.999),
    }
    for label, selection in selections.items():
        if np.sum(selection) < 100:
            continue
        result = {"drug": "norepinephrine", "outcome": events[channel],
                  "horizon": horizon, "subset": label,
                  **orthogonal_coefficient_v2(rA, rY, selection)}
        ne_results_v2.append(result)
        print(json.dumps(result), flush=True)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Strictly pre-dose norepinephrine checks\n"
            "Three stay-level folds. Rate and outcome nuisance models fit on matching "
            "positive-rate, observed-outcome rows. Prefix-only histories; no future covariates. "
            "Checking early versus late hours distinguishes the previously observed early-stay "
            "long-horizon association from effects relevant to the rollout.\n"
            + json.dumps(ne_results_v2) + "\n")
print("Completed in", round(time.time() - analysis_started, 1), "seconds")
```
