The nonlinear estimates support roughly −2 glucose units per insulin rate unit and +3.3 MAP units per norepinephrine rate unit for the next reading. Their consistency is encouraging, but it does not yet establish the 12-hour response. I’ll estimate longer-lag responses while accounting for subsequent dose changes. A coherent decay pattern would support a rollout target; erratic estimates would weaken it.

```python
from scipy.optimize import least_squares

print("=== FORWARD CSV SCHEMAS ===")
for path in ["/work/fwd_1/factual_original.csv", "/work/fwd_2/dose_edit_original.csv"]:
    sample = pd.read_csv(path)
    print(path, sample.shape, sample.columns.tolist())
    print(sample.head(3).to_string(index=False))

def crossfit_predict(X, target, folds, valid, max_iter=130):
    prediction = np.full(len(target), np.nan)
    for fold in (0, 1):
        tr = valid & (folds != fold)
        va = valid & (folds == fold)
        model = HistGradientBoostingRegressor(
            max_iter=max_iter, max_leaf_nodes=31, learning_rate=.07,
            min_samples_leaf=120, l2_regularization=10.0,
            early_stopping=False, random_state=321
        )
        model.fit(X[tr], target[tr])
        prediction[va] = model.predict(X[va])
    return prediction

impulse_results = {}
impulse_cache = {}
start = time.time()

for outcome, drug in [("glucose_finger", "insulin"), ("map_arterial", "norepinephrine")]:
    cache = dml_cache[drug]
    inds, X, groups = cache["inds"], cache["X"], cache["groups"]
    folds = ((groups - train_ids[0]) % 2).astype(int)
    hours, rd = cache["hours"], cache["rd"]
    current = wide[outcome+"_last"].iloc[inds].to_numpy()
    future_doses = [cache["d"]]
    residual_future_doses = [rd]
    for j in range(1, 12):
        future = wide[drug].groupby(level=0).shift(-j).iloc[inds].to_numpy()
        valid = np.isfinite(future) & (hours + j <= 23)
        pred = crossfit_predict(X, future, folds, valid, max_iter=100)
        future_doses.append(future)
        residual_future_doses.append(future - pred)

    kernel, rows, stored = [], [], []
    for L in range(1, 13):
        future_y = wide[outcome].groupby(level=0).shift(-L).iloc[inds].to_numpy()
        target = future_y - current
        valid = np.isfinite(target) & (hours + L <= 24)
        for j in range(L):
            valid &= np.isfinite(residual_future_doses[j])
        pred = crossfit_predict(X, target, folds, valid)
        rr = rd[valid]
        residual_y = target[valid] - pred[valid]
        den = rr @ rr
        total = (rr @ residual_y) / den
        feedback = np.array([
            (rr @ residual_future_doses[j][valid]) / den for j in range(L)
        ])
        controlled = total - sum(
            kernel[L-j-1] * feedback[j] for j in range(1, L)
        )
        kernel.append(float(controlled))
        approximate_se = residual_effect(residual_y, rr, groups[valid])[1]
        rows.append({
            "lag": L, "total_LP": round(float(total), 4),
            "controlled_impulse": round(float(controlled), 4),
            "total_LP_SE": approximate_se,
            "next_dose_feedback": round(float(feedback[1]), 4) if L > 1 else 0,
            "n": int(valid.sum())
        })
        stored.append(dict(valid=valid, residual_y=residual_y, feedback=feedback))
    lag = np.arange(12, dtype=float)
    b = np.array(kernel)
    weights = 1 / np.maximum(np.array([r["total_LP_SE"] for r in rows]), .05)
    fit = least_squares(
        lambda pars: weights * (pars[0] * pars[1]**lag - b),
        x0=[b[0], .85], bounds=([-20, 0], [20, 1.15])
    )
    print("\n", drug)
    print(pd.DataFrame(rows).to_string(index=False))
    print("Geometric impulse fit [first-hour effect, persistence]:", fit.x.round(5))
    impulse_results[drug] = {
        "rows": rows, "geometric": fit.x.tolist(), "kernel": kernel
    }
    impulse_cache[drug] = stored

print("\nElapsed:", round(time.time()-start, 1), "seconds")
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 5\n"
        "Estimated 1..12 hour local projections using held-out history models. "
        "Subtracted subsequent-treatment feedback using future-dose residual "
        "covariances to estimate controlled impulse responses. Geometric "
        "summaries are tentative; long-lag selection and recursive estimation "
        "can amplify error. No weight edits yet.\n"
        + json.dumps(impulse_results, indent=2) + "\n"
    )
```