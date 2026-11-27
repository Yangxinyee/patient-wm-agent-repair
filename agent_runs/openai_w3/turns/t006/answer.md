History adjustment finds an insulin-associated decrease of about 1.91 glucose units per unit/hour and a norepinephrine-associated increase of about 3.23 MAP units per rate unit. These differ substantially from the model’s responses, but are not yet definitive causal estimates. I’ll test stronger adjustment, alternate sensors, and dose nonlinearity. Stable estimates would support a repair; sensitivity would warn against one.

```python
started = time.time()
robust_results = {}

for drug, outcome in [("insulin", "glucose_finger"),
                      ("insulin", "glucose_lab"),
                      ("norepinephrine", "map_arterial"),
                      ("norepinephrine", "map_cuff")]:
    dose = A[drug][:, 1:25].reshape(-1)
    next_y = A[outcome][:, 2:26].reshape(-1)
    rows = np.flatnonzero(np.isfinite(next_y))
    xx, dd, yy = Xhist[rows], dose[rows], next_y[rows]
    predictions = np.zeros((len(rows), 3))
    for fold in [0, 1]:
        fit = fold_rows[rows] != fold
        test = ~fit
        params = dict(max_iter=350, max_leaf_nodes=31, learning_rate=.07,
                      max_bins=192, min_samples_leaf=100, l2_regularization=8,
                      early_stopping=False, random_state=197 + fold)
        for j, target in enumerate([dd, dd*dd, yy]):
            model = HistGradientBoostingRegressor(**params).fit(xx[fit], target[fit])
            predictions[test, j] = model.predict(xx[test])
    rd = dd - predictions[:, 0]
    r2 = dd*dd - predictions[:, 1]
    ry = yy - predictions[:, 2]
    beta, se = cluster_estimate(rd, ry, sid_rows[rows])
    rr = np.column_stack([rd, r2])
    quadratic = np.linalg.lstsq(rr, ry, rcond=None)[0]
    bread = np.linalg.inv(rr.T @ rr)
    errors = ry - rr @ quadratic
    scores = np.zeros((n_stays, 2))
    np.add.at(scores, sid_rows[rows], rr * errors[:, None])
    qse = np.sqrt(np.diag(bread @ (scores.T @ scores) @ bread))
    robust_results[(drug, outcome)] = dict(
        rows=rows, pred_d=predictions[:, 0], pred_y=predictions[:, 2],
        rd=rd, ry=ry, beta=beta, se=se,
        quadratic=quadratic, quadratic_se=qse
    )
    print("\n", drug, "->", outcome, "n =", len(rows))
    print("Linear effect:", round(beta, 5), "cluster SE:", round(se, 5))
    print("Quadratic coefficients [dose, dose^2]:", quadratic.round(5).tolist(),
          "SE:", qse.round(5).tolist())
    print("Dose R2:", round(1 - np.mean(rd**2)/np.var(dd), 5))
    # Stratify by dose predicted FROM HISTORY, rather than by realized treatment.
    edges = [0, .75, 1.5, 2.5, np.inf] if drug == "insulin" else [0, .15, .4, .7, np.inf]
    pdose = predictions[:, 0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (pdose >= lo) & (pdose < hi)
        if mask.sum() > 1000:
            b, s = cluster_estimate(rd[mask], ry[mask], sid_rows[rows][mask])
            print("history-predicted dose", lo, "to", hi,
                  "n:", int(mask.sum()), "effect:", round(b, 4), "SE:", round(s, 4))
    paired = ("glucose_lab" if outcome == "glucose_finger" else
              "glucose_finger" if outcome == "glucose_lab" else
              "map_cuff" if outcome == "map_arterial" else "map_arterial")
    both_now = np.isfinite(A[outcome][:, 1:25].reshape(-1)[rows]) & \
               np.isfinite(A[paired][:, 1:25].reshape(-1)[rows])
    if both_now.sum() > 1000:
        b, s = cluster_estimate(rd[both_now], ry[both_now], sid_rows[rows][both_now])
        print("Both sensors observed before dose:", int(both_now.sum()),
              "effect:", round(b, 4), "SE:", round(s, 4))

print("\nElapsed seconds:", round(time.time() - started, 1))
summary = {str(k): {"beta": v["beta"], "se": v["se"],
                   "quadratic": v["quadratic"].tolist(),
                   "quadratic_se": v["quadratic_se"].tolist()}
           for k, v in robust_results.items()}
with Path("/work/notes.md").open("a") as f:
    f.write("\nPython runs: 5; forwards: 1. Stronger 350-iteration, twofold "
            "history-adjusted estimates on all four sensors; also residualized "
            "dose-squared to check nonlinearity. " + str(summary) + "\n")
```
