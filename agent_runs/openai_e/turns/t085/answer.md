The panel separates pre-dose readings from next-hour outcomes, and the norepinephrine policy is strongly nonlinear in MAP. I’ll estimate responses to residual dose innovations through eight hours, including the later doses those innovations induce. A common impulse-response curve would support a persistence estimate; a poor fit would refute the simple recurrence assumption.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

common73 = s69["t"].between(0, 16).to_numpy()
groups73 = s69["stay_id"].to_numpy()
settings73 = dict(settings69)
settings73["max_iter"] = min(int(settings73.get("max_iter", 180)), 180)
print("Nuisance settings:", settings73)
print("Feature names:", features69[1])

def crossfit_ne73(y):
    y = np.asarray(y, dtype=float)
    prediction = np.full(len(y), np.nan)
    models = []
    for fold in [0, 1]:
        train = common73 & (fold69 != fold) & np.isfinite(y)
        test = common73 & (fold69 == fold)
        model = HistGradientBoostingRegressor(**settings73)
        model.fit(X69[train], y[train])
        prediction[test] = model.predict(X69[test])
        models.append(model)
    return prediction, models

def moment_ne73(y, prediction, mask):
    mask = mask & np.isfinite(y) & np.isfinite(prediction)
    z = resid_d69[mask]
    residual = np.asarray(y)[mask] - np.asarray(prediction)[mask]
    denominator = np.dot(z, z)
    beta = np.dot(z, residual) / denominator
    ids, inverse = np.unique(groups73[mask], return_inverse=True)
    influence = np.bincount(inverse, weights=z * (residual - beta * z))
    se = np.sqrt(np.sum(influence**2) * len(ids) / (len(ids) - 1)) / denominator
    return {"n": int(mask.sum()), "beta": float(beta), "se": float(se)}, mask

ne_future_d73 = {0: d69}
ne_future_pred_d73 = {0: pred_d69}
ne_future_models73 = {}
for j in range(1, 8):
    y = panel.groupby("stay_id", sort=False)["norepinephrine"].shift(-j).reindex(s69.index).to_numpy()
    ne_future_d73[j] = y
    ne_future_pred_d73[j], ne_future_models73[("dose", j)] = crossfit_ne73(y)

ne_horizons73 = [1, 2, 3, 4, 6, 8]
ne_future_y73 = {}
ne_future_pred_y73 = {}
ne_dynamic73 = {}
ne_G73 = {}
for h in ne_horizons73:
    y = panel.groupby("stay_id", sort=False)["map_any"].shift(-h).reindex(s69.index).to_numpy()
    ne_future_y73[h] = y
    if h == 1:
        prediction = pred_y69["map_any_next"]
    else:
        prediction, models = crossfit_ne73(y)
        ne_future_models73[("map_any", h)] = models
    ne_future_pred_y73[h] = prediction
    estimate, mask = moment_ne73(y, prediction, common73)
    ne_dynamic73[h] = estimate
    z = resid_d69[mask]
    denominator = np.dot(z, z)
    ne_G73[h] = np.array([
        np.dot(z, (ne_future_d73[j] - ne_future_pred_d73[j])[mask]) / denominator
        for j in range(h)
    ])

print("\nInnovation response and induced-dose moments")
for h in ne_horizons73:
    print(h, ne_dynamic73[h], "dose moments", np.round(ne_G73[h], 4).tolist())

phi_grid73 = np.linspace(-0.5, 1.05, 1551)
b73 = np.array([ne_dynamic73[h]["beta"] for h in ne_horizons73])
se73 = np.array([ne_dynamic73[h]["se"] for h in ne_horizons73])
profile73 = []
for phi in phi_grid73:
    response = np.array([
        np.dot(ne_G73[h], phi ** np.arange(h - 1, -1, -1))
        for h in ne_horizons73
    ])
    weight = 1 / np.maximum(se73, 0.05)**2
    beta = np.dot(weight * response, b73) / np.dot(weight * response, response)
    loss = np.sum(weight * (b73 - beta * response)**2)
    profile73.append((loss, beta, phi))
best_ne73 = min(profile73)
print("\nJoint fit: loss, beta, persistence =", best_ne73)
phi_ne73 = best_ne73[2]
beta_ne73 = best_ne73[1]
print("Fitted reduced-form curve:", [
    round(beta_ne73 * np.dot(ne_G73[h], phi_ne73 ** np.arange(h - 1, -1, -1)), 4)
    for h in ne_horizons73
])
print("Provisional rollout target:", record_schedule_target69(
    beta_ne73, phi_ne73, "norepinephrine", "norepinephrine_x2"
))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 73: dynamic NE innovation identification\n"
            "Cross-fitted future MAP and future dose nuisance models on a common t<=16 cohort. "
            "Jointly fit one-hour effect and persistence to MAP innovation moments, accounting "
            "for induced subsequent doses. "
            f"Moments={ne_dynamic73}; dose moments={ {h: ne_G73[h].tolist() for h in ne_horizons73} }; "
            f"joint fit={best_ne73}. These are provisional pending separate arterial/cuff and "
            "placebo checks. No weights changed. Python runs: 73; forwards: 10.\n")
```
