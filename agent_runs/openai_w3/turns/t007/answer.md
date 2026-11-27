Insulin’s estimate is stable across both glucose sensors and consistent with a linear effect near −2. Norepinephrine is positive in well-supported dose ranges, but low-dose and cuff estimates show residual confounding. I’ll now estimate how effects persist while accounting for later doses. A stable decay pattern across sensors would support extrapolating to the rollout readouts.

```python
started = time.time()
print("=== FORWARD FILE FORMAT ===")
baseline_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(baseline_rollout.shape, baseline_rollout.columns.tolist())
print(baseline_rollout.head(5).to_string(index=False))

# Treatment innovations estimated from all pre-dose histories.
dose_predictions_full = {}
for drug in events6[-2:]:
    dd = A[drug][:, 1:25].reshape(-1)
    pred = np.zeros(len(dd))
    for fold in [0, 1]:
        fit = fold_rows != fold
        model = HistGradientBoostingRegressor(
            max_iter=350, max_leaf_nodes=31, learning_rate=.07,
            max_bins=192, min_samples_leaf=100, l2_regularization=8,
            early_stopping=False, random_state=297 + fold
        ).fit(Xhist[fit], dd[fit])
        pred[~fit] = model.predict(Xhist[~fit])
    dose_predictions_full[drug] = pred

# Structural nested/local-projection calculation:
# remove the estimated contributions of doses AFTER the baseline dose,
# then use its history-adjusted innovation to estimate its remaining effect.
impulse_results = {}
for drug, outcome in [("insulin", "glucose_finger"),
                      ("insulin", "glucose_lab"),
                      ("norepinephrine", "map_arterial")]:
    theta, theta_se, unadjusted = [], [], []
    print("\n=== CONTROLLED IMPULSE:", drug, "->", outcome, "===")
    for horizon in range(1, 13):
        width = 25 - horizon
        base_rows = (np.arange(n_stays)[:, None] * 24 +
                     np.arange(width)[None, :]).reshape(-1)
        yy = A[outcome][:, 1+horizon:26].reshape(-1).copy()
        adjusted = yy.copy()
        for j in range(1, horizon):
            adjusted -= theta[horizon-j-1] * A[drug][:, 1+j:1+j+width].reshape(-1)
        good = np.isfinite(adjusted)
        rows = base_rows[good]
        target = adjusted[good]
        xx = Xhist[rows]
        predictions = np.zeros(len(rows))
        for fold in [0, 1]:
            fit = fold_rows[rows] != fold
            model = HistGradientBoostingRegressor(
                max_iter=220, max_leaf_nodes=31, learning_rate=.075,
                max_bins=128, min_samples_leaf=120, l2_regularization=8,
                early_stopping=False, random_state=397 + fold
            ).fit(xx[fit], target[fit])
            predictions[~fit] = model.predict(xx[~fit])
        dd = A[drug][:, 1:25].reshape(-1)[rows]
        rd = dd - dose_predictions_full[drug][rows]
        ry = target - predictions
        b, s = cluster_estimate(rd, ry, sid_rows[rows])
        theta.append(b)
        theta_se.append(s)
        print("h =", horizon, "effect =", round(b, 5),
              "SE (conditional on earlier estimates) =", round(s, 5))
    impulse_results[(drug, outcome)] = {
        "theta": np.array(theta), "se": np.array(theta_se)
    }

print("\n=== HEAD/CONTEXT TENSOR NAMES ===")
for name in NAMES:
    if name.startswith(("core.z_", "core.gap", "core.out_norm",
                        "z_head", "gap.", "filter", "last_numeric", "type_emb")):
        print(name, SHAPES[name])
print("numeric_note:", VOCAB.get("numeric_note", ""))

print("\nElapsed seconds:", round(time.time() - started, 1))
summary = {str(k): v["theta"].round(6).tolist() for k, v in impulse_results.items()}
with Path("/work/notes.md").open("a") as f:
    f.write("\nPython runs: 6; forward requests: 1. Robust immediate estimates: "
            "insulin finger -1.9369, lab -1.9077; little quadratic evidence. "
            "NE arterial +3.192, paired pre-dose sensors +3.329, high expected "
            "rates +3.407; cuff +2.693 with confounding near treatment onset. "
            "Estimated controlled impulse kernels by sequentially removing later "
            "dose contributions, using cross-fitted baseline treatment innovations. "
            "Kernels: " + str(summary) + "\n")
```
