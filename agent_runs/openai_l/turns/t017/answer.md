The recovered variables include cross-fitted dose and outcome residuals, but not their final estimates. I’ll recompute stay-clustered effects and inspect norepinephrine’s dosing rule. Stable estimates across sensors and observation subsets would support them; large differences would indicate remaining confounding or selection.

```python
import pandas as pd
print("Panel hours:", rec_hours)
print("Panel shapes:", {k: v.shape for k, v in A.items()})
print("History features:", hist_names)

def clustered_residual_fit(X, y, mask, groups):
    ok = mask & np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    xx = X[ok].astype(np.float64)
    yy = y[ok].astype(np.float64)
    gg = groups[ok]
    inv = np.linalg.pinv(xx.T @ xx)
    bb = inv @ (xx.T @ yy)
    err = yy - xx @ bb
    scores = np.column_stack([
        np.bincount(gg, weights=xx[:, j] * err, minlength=30000)
        for j in range(xx.shape[1])
    ])
    cov = inv @ (scores.T @ scores) @ inv
    return int(ok.sum()), bb, np.sqrt(np.maximum(np.diag(cov), 0))

DX = np.column_stack([dml_dose_resid["insulin"], dml_dose_resid["norepinephrine"]])
groups14 = np.repeat(np.arange(30000), 24)
effects14 = {}
for drug14, sensors14 in [
    ("insulin", ("glucose_finger", "glucose_lab")),
    ("norepinephrine", ("map_arterial", "map_cuff"))
]:
    ai14 = A[drug14][:, 1:25].reshape(-1)
    c1 = np.isfinite(P[sensors14[0]][:, 1:25]).reshape(-1)
    c2 = np.isfinite(P[sensors14[1]][:, 1:25]).reshape(-1)
    masks14 = {
        "all": np.ones(len(ai14), bool),
        "active": ai14 > 0,
        "current own": None,
        "active, both current": (ai14 > 0) & c1 & c2
    }
    for sensor14 in sensors14:
        masks14["current own"] = c1 if sensor14 == sensors14[0] else c2
        for label14, mask14 in masks14.items():
            n14, b14, se14 = clustered_residual_fit(
                DX, dml_outcome_resid[sensor14], mask14, groups14)
            effects14[(sensor14, label14)] = (n14, b14, se14)
            print(sensor14, label14, "n", n14,
                  "I,NE", np.round(b14, 4), "SE", np.round(se14, 4))

print("\n=== PREVIOUS THRESHOLD RESULTS (compact) ===")
for r14 in rd_results:
    text14 = str(r14)
    if "179.5" in text14 or "219.5" in text14:
        print(text14[:700])

print("\n=== NE DOSE VS LAST ARTERIAL READING ===")
ne14 = A["norepinephrine"][:, 1:25].reshape(-1)
ma14 = Last["map_arterial"][:, 1:25].reshape(-1)
policy14 = pd.DataFrame({"MAPbin": np.floor(ma14 / 2) * 2, "rate": ne14})
print(policy14.groupby("MAPbin").rate.agg(
    n="size", mean="mean", median="median",
    active=lambda x: float((x > 0).mean())
).loc[40:90].round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 14\nRecomputed cross-fitted two-treatment residual regressions with stay-clustered uncertainty and observation/active-treatment subsets; retained results in effects14. Inspecting NE policy support before accepting its effect.\n")
```
