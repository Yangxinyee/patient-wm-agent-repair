The original factual NLL is 1.63461 on these 100 stays. Norepinephrine’s effect remains unresolved: ordinary adjustment is sensitive to history, and the sensor-disagreement instrument is weak. I’ll compare measurement-error-corrected and lag-instrumented dynamics. Agreement across these methods would support an effect estimate; disagreement would leave the target uncertain.

```python
factual_baseline = pd.read_csv("/work/fwd_2/factual_original.csv")
print("=== FACTUAL FILE ===")
print(factual_baseline.head(3).to_string(index=False))
print("Columns:", factual_baseline.columns.tolist())

def iv_fit(Y, R, Q, X, ids):
    Y = np.asarray(Y, float)
    R = np.asarray(R, float)
    Q = np.asarray(Q, float)
    combined = np.column_stack([Y, R, Q])
    combined -= X @ lstsq(X, combined, lapack_driver="gelsy")[0]
    y = combined[:, 0]
    r = combined[:, 1:1+R.shape[1]]
    q = combined[:, 1+R.shape[1]:]
    qq_inv = np.linalg.pinv(q.T@q)
    rq = r.T@q
    A = rq@qq_inv@rq.T
    beta = np.linalg.pinv(A) @ (rq@qq_inv@(q.T@y))
    error = y-r@beta
    groups, unique_ids = pd.factorize(ids, sort=False)
    scores = np.column_stack([
        np.bincount(groups, weights=q[:, j]*error)
        for j in range(q.shape[1])])
    bread = np.linalg.pinv(A)@rq@qq_inv
    cov = bread@(scores.T@scores)@bread.T
    return beta, np.sqrt(np.maximum(np.diag(cov), 0)), np.linalg.cond(A)

def corrected_level(outcome, proxy, drug, mode, interior=False):
    y = pgroup[outcome].shift(-1).to_numpy().astype(float)
    current = panel[outcome].to_numpy().astype(float)
    other = panel[proxy].to_numpy().astype(float)
    d = panel[drug].to_numpy().astype(float)
    valid = ((time_array >= 0) & (time_array < 24) &
             np.isfinite(y) & np.isfinite(current) & np.isfinite(other))
    if interior:
        valid &= (d > 0) & (d < (1 if drug == "norepinephrine" else 5))
    ix = np.flatnonzero(valid)
    if mode == "basic":
        X = np.column_stack([np.ones(len(ix)), time_array[ix]/24,
                             (time_array[ix]/24)**2])
    else:
        keep = [j for j, name in enumerate(history_names)
                if not (name.startswith(outcome) or name.startswith(proxy))
                or "_lag" in name]
        X = np.column_stack([np.ones(len(ix)), HX[ix][:, keep]])
    R = np.column_stack([current[ix], d[ix]])
    Q = np.column_stack([other[ix], d[ix]])
    Y = y[ix].copy()
    if mode == "stayFE":
        groups, _ = pd.factorize(stay_array[ix], sort=False)
        counts = np.bincount(groups)
        for arr in [X, R, Q, Y[:, None]]:
            for j in range(arr.shape[1]):
                arr[:, j] -= (np.bincount(groups, weights=arr[:, j])/counts)[groups]
    b, se, condition = iv_fit(Y, R, Q, X, stay_array[ix])
    print(drug, outcome, mode, "interior:", interior, "n:", len(ix),
          "persistence:", round(b[0], 4), "+/-", round(se[0], 4),
          "dose effect:", round(b[1], 4), "+/-", round(se[1], 4))
    return {"persistence": b[0], "effect": b[1],
            "se_persistence": se[0], "se_effect": se[1], "n": len(ix)}

def difference_dynamics(outcome, proxy, drug):
    current = panel[outcome].to_numpy().astype(float)
    previous = pgroup[outcome].shift(1).to_numpy().astype(float)
    future = pgroup[outcome].shift(-1).to_numpy().astype(float)
    d = panel[drug].to_numpy().astype(float)
    d1 = pgroup[drug].shift(1).to_numpy().astype(float)
    instruments = []
    for lag in [1, 2, 3]:
        instruments.append(pgroup[proxy].shift(lag).to_numpy().astype(float))
        instruments.append(pgroup[drug].shift(lag).to_numpy().astype(float))
    Q = np.column_stack(instruments)
    valid = ((time_array >= 3) & (time_array < 24) &
             np.isfinite(current) & np.isfinite(previous) & np.isfinite(future) &
             np.isfinite(Q).all(axis=1))
    ix = np.flatnonzero(valid)
    Y = (future-current)[ix]
    R = np.column_stack([(current-previous)[ix], (d-d1)[ix]])
    X = np.column_stack([np.ones(len(ix)), time_array[ix]/24])
    b, se, condition = iv_fit(Y, R, Q[ix], X, stay_array[ix])
    print(drug, outcome, "first-difference IV", "n:", len(ix),
          "persistence:", round(b[0], 4), "+/-", round(se[0], 4),
          "dose effect:", round(b[1], 4), "+/-", round(se[1], 4),
          "condition:", round(condition, 1))
    return {"persistence": b[0], "effect": b[1],
            "se_persistence": se[0], "se_effect": se[1], "n": len(ix)}

print("\n=== MEASUREMENT-ERROR-CORRECTED LEVEL DYNAMICS ===")
dynamic_results = {}
for outcome, proxy, drug in [
    ("map_arterial", "map_cuff", "norepinephrine"),
    ("map_cuff", "map_arterial", "norepinephrine"),
    ("glucose_finger", "glucose_lab", "insulin"),
]:
    for mode in ["basic", "history", "stayFE"]:
        dynamic_results[(outcome, drug, mode)] = corrected_level(
            outcome, proxy, drug, mode)
    dynamic_results[(outcome, drug, "history_interior")] = corrected_level(
        outcome, proxy, drug, "history", interior=True)

print("\n=== LAG-INSTRUMENTED FIRST-DIFFERENCE DYNAMICS ===")
for outcome, proxy, drug in [
    ("map_arterial", "map_cuff", "norepinephrine"),
    ("map_cuff", "map_arterial", "norepinephrine"),
    ("glucose_finger", "glucose_lab", "insulin"),
]:
    dynamic_results[(outcome, drug, "differenceIV")] = difference_dynamics(outcome, proxy, drug)

with open("/work/notes.md", "a") as f:
    f.write("\n## Factual baseline and dynamic identification; Python 5, forward 2\n")
    f.write("- Original factual value NLL = 1.63461 on first 100 tuning stays, 4751 readings.\n")
    f.write("- NE sensor-difference first stage is weak; do not use its point estimates.\n")
    f.write("- Tested replicate-sensor corrections and lag-instrumented first differences; "
            "each relies on explicit measurement-noise/innovation assumptions.\n")
    f.write(json.dumps({str(k): v for k, v in dynamic_results.items()}, indent=2)+"\n")
```