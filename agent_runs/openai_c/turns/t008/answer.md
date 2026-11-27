The state capture succeeded. One-hour effects are not enough to set the rollout targets: effects can persist, and clinicians may adjust later doses. I’ll estimate persistence while accounting for those later dose changes, then calculate the four shifts under the evaluation’s fixed-dose edits. Consistent horizon estimates would support this approximation; poor fits would refute it.

```python
PYTHON_RUNS += 1
FORWARD_PASSES = 3
from scipy.optimize import least_squares

state_file = np.load("/work/fwd_3/states_original.npz")
states_original = {k: state_file[k] for k in state_file.files}
print("=== STATE ARRAYS ===")
for k, a in states_original.items():
    print(k, a.shape, a.dtype)

print("\n=== ROLLOUT FILE FORMAT ===")
original_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(original_rollout.head(3).to_string(index=False))

def insulin_rd_path(horizon, bandwidth=14):
    running = panel.glucose_finger.to_numpy()
    cutoffs = np.array([150., 180., 220., 260., 300.])
    nearest = np.argmin(abs(running[:, None] - (cutoffs - .5)), axis=1)
    r = running - (cutoffs[nearest] - .5)
    future = g.glucose_finger.shift(-horizon).to_numpy()
    mask = (np.isfinite(running) & np.isfinite(future) &
            (abs(r) < bandwidth) & (t >= 0) & (t <= 24-horizon))
    rr = r[mask]
    above = (rr >= 0).astype(float)
    indicators = np.eye(5)[nearest[mask]]
    names = [c for c in controls.columns
             if c not in ["glucose_finger", "glucose_finger_square"]]
    H = controls.loc[mask, names].to_numpy()
    H = (H-H.mean(0)) / np.maximum(H.std(0), 1e-8)
    C = np.column_stack([
        indicators, indicators*rr[:, None],
        indicators*(rr*above)[:, None], t[mask], H
    ])
    columns = [above, panel.insulin.to_numpy()[mask], future[mask]]
    for j in range(1, horizon):
        columns.append(g.insulin.shift(-j).to_numpy()[mask])
    w = np.sqrt(1-abs(rr)/bandwidth)
    C *= w[:, None]
    R = np.column_stack(columns)*w[:, None]
    R -= C @ np.linalg.lstsq(C, R, rcond=None)[0]
    z, d, y = R[:, 0], R[:, 1], R[:, 2]
    effect, se = clustered_ratio(z, d, y, sid[mask])
    denom = np.dot(z, d)
    phi = [1.] + [float(np.dot(z, R[:, 2+j])/denom)
                  for j in range(1, horizon)]
    return dict(h=horizon, n=int(mask.sum()), effect=effect, se=se, phi=phi)

def crossfit_residual_matrix(X, Y, groups):
    prediction = np.zeros_like(Y, dtype=float)
    fold_ids = (groups-1010000000) % 2
    for fold in [0, 1]:
        tr = np.flatnonzero(fold_ids != fold)
        te = np.flatnonzero(fold_ids == fold)
        if len(tr) > 140000:
            tr = np.random.default_rng(430+fold).choice(tr, 140000, replace=False)
        for j in range(Y.shape[1]):
            reg = HistGradientBoostingRegressor(
                max_iter=180, learning_rate=.08, max_leaf_nodes=31,
                min_samples_leaf=75, l2_regularization=10,
                max_bins=128, random_state=40+fold)
            reg.fit(X[tr], Y[tr, j])
            prediction[te, j] = reg.predict(X[te])
    return Y-prediction

def ne_dml_path(horizon):
    future = g.map_arterial.shift(-horizon)
    mask = (future.notna() & panel.map_arterial.notna() &
            (t >= 0) & (t <= 24-horizon))
    Xdf = controls.loc[mask].copy()
    Xdf["other_current_drug"] = panel.loc[mask, "insulin"]
    X = Xdf.to_numpy(dtype=np.float32)
    columns = [panel.loc[mask, "norepinephrine"].to_numpy(),
               future[mask].to_numpy()]
    for j in range(1, horizon):
        columns.append(g.norepinephrine.shift(-j)[mask].to_numpy())
    R = crossfit_residual_matrix(X, np.column_stack(columns), sid[mask])
    z, y = R[:, 0], R[:, 1]
    effect, se = clustered_ratio(z, z, y, sid[mask])
    denom = np.dot(z, z)
    phi = [1.] + [float(np.dot(z, R[:, 1+j])/denom)
                  for j in range(1, horizon)]
    return dict(h=horizon, n=int(mask.sum()), effect=effect, se=se, phi=phi)

print("\n=== HORIZON EFFECTS AND SUBSEQUENT DOSE RESPONSES ===")
dynamic_moments = {"insulin": [], "norepinephrine": []}
start = time.time()
for h in [1, 2, 4, 6]:
    for drug, fn in [("insulin", insulin_rd_path),
                     ("norepinephrine", ne_dml_path)]:
        result = fn(h)
        dynamic_moments[drug].append(result)
        print(drug, "h", h, "n", result["n"],
              "effect/SE", np.round([result["effect"], result["se"]], 4),
              "future-dose response", np.round(result["phi"], 4))

def fit_persistence(drug):
    moments = dynamic_moments[drug]
    def errors(p):
        beta, rho = p
        pred = [beta*sum(rho**(m["h"]-1-j)*v
                         for j, v in enumerate(m["phi"])) for m in moments]
        return np.array([(p-m["effect"])/max(m["se"], .05)
                         for p, m in zip(pred, moments)])
    if drug == "insulin":
        initial, lower, upper = [-2., .9], [-8., 0.], [0., 1.04]
    else:
        initial, lower, upper = [3., .9], [0., 0.], [12., 1.04]
    fit = least_squares(errors, initial, bounds=(lower, upper))
    beta, rho = fit.x
    pred = [beta*sum(rho**(m["h"]-1-j)*v
                     for j, v in enumerate(m["phi"])) for m in moments]
    return dict(beta=float(beta), rho=float(rho),
                observed=[m["effect"] for m in moments],
                fitted=[float(v) for v in pred],
                standardized_residuals=errors(fit.x).tolist())

persistence = {drug: fit_persistence(drug) for drug in DRUGS}
print("\n=== PERSISTENCE FITS ===")
for drug, result in persistence.items():
    print(drug, result)

def record_based_rollouts(p, ids=None, parameters=None):
    parameters = persistence if parameters is None else parameters
    if ids is not None:
        p = p.loc[p.index.get_level_values("stay_id").isin(ids)]
    result = {}
    for drug, edits in [("insulin", [("insulin_x2", 2.), ("insulin_half", .5)]),
                        ("norepinephrine", [("ne_x2", 2.), ("ne_half", .5)])]:
        rates = p[drug].unstack("t").reindex(columns=np.arange(-1, 25)).to_numpy()
        beta, rho = parameters[drug]["beta"], parameters[drug]["rho"]
        for edit, multiplier in edits:
            shift = np.zeros_like(rates)
            reached = np.zeros_like(rates, dtype=bool)
            for hour in range(0, 24):
                cur, nxt = hour+1, hour+2
                dose_change = ((multiplier-1)*rates[:, cur] if hour >= 12 else 0)
                shift[:, nxt] = rho*shift[:, cur] + beta*dose_change
                reached[:, nxt] = reached[:, cur] | (
                    (rates[:, cur] > 0) if hour >= 12 else False)
            use = reached[:, 14:26]  # hours 13..24
            result[edit] = dict(mean=float(shift[:, 14:26][use].mean()),
                                n_stay_hours=int(use.sum()),
                                n_stays=int(use.any(1).sum()))
    return result

target_probe = record_based_rollouts(panel_tuning, probe_ids)
target_train = record_based_rollouts(panel)
print("\n=== RECORD-BASED ROLLOUT TARGETS (linear persistence approximation) ===")
print("Probe:", target_probe)
print("Training population:", target_train)
print("Elapsed seconds:", round(time.time()-start, 1))

print("\n=== RELEVANT CORE TENSOR NAMES ===")
for name in NAMES:
    if any(s in name for s in ["z_post", "z_prior", "z_in", "z_head",
                               "prelude_norm", "c_out", "out_norm"]):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 5 / forward 3: persistence and state capture\n"
        "- Original states obtained on training IDs 1010000000..1010000099, "
        "all bundles (4035 rows; 992-dimensional state).\n"
        f"- Dynamic moments: {dynamic_moments!r}\n"
        f"- Persistence fit accounting for later treatment responses: {persistence!r}\n"
        f"- Approximate fixed-dose-edit targets, probe: {target_probe!r}\n"
        f"- Approximate targets, training population: {target_train!r}\n"
        "- Targets remain model-based estimates from records, not observed "
        "counterfactual truth. Check horizon-fit residuals and method agreement.\n"
        "- Python runs: 5; forward passes: 3; no edits.\n"
    )
```
