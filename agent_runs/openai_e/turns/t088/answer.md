The corrected moments support a positive effect near 3 MAP units per dose unit and persistence near 0.70. Accounting for subsequent dosing gives provisional targets of +4.22 for norepinephrine x2 and −2.11 for half-dose. I’ll check arterial and cuff readings separately, pre-treatment placebos, and stay-cluster uncertainty. Large disagreements would undermine this target.

```python
def fit_impulse76(horizons, estimates, dose_moments, grid=phi_grid75):
    b = np.array([estimates[h]["beta"] for h in horizons])
    se = np.array([estimates[h]["se"] for h in horizons])
    weight = 1 / np.maximum(se, 0.05)**2
    response = np.stack([
        np.sum(dose_moments[h][:, None] *
               grid[None, :] ** np.arange(h - 1, -1, -1)[:, None], axis=0)
        for h in horizons
    ])
    numerator = np.sum(weight[:, None] * response * b[:, None], axis=0)
    denominator = np.sum(weight[:, None] * response**2, axis=0)
    beta = numerator / denominator
    loss = np.sum(weight * b**2) - numerator**2 / denominator
    k = int(np.argmin(loss))
    return {"loss": float(loss[k]), "beta": float(beta[k]), "phi": float(grid[k])}

ne_checks76 = {}
for event in ["map_arterial", "map_cuff"]:
    estimates, moments = {}, {}
    for h in ne_horizons75:
        y = aligned_future74(event, h)
        prediction, models = crossfit_ne73(y)
        ne_models75[(event, h)] = models
        estimate, mask = moment_ne75(y, prediction, common73)
        estimates[h] = estimate
        z = ne_z75[mask]
        moments[h] = np.array([
            np.dot(z, (ne_future_d75[j] - ne_future_pred_d75[j])[mask]) / np.dot(z, z)
            for j in range(h)
        ])
    fit = fit_impulse76(ne_horizons75, estimates, moments)
    target = record_schedule_target69(fit["beta"], fit["phi"], "norepinephrine", "ne_x2")
    ne_checks76[event] = {"estimates": estimates, "fit": fit, "target": target}
    print(event, "moments", {
        h: (round(estimates[h]["beta"], 3), round(estimates[h]["se"], 3))
        for h in ne_horizons75
    })
    print("fit", fit, "x2 target", target)

ne_placebos76 = {}
for lag in [5, 8]:
    y = aligned_future74("map_any", -lag)
    prediction, models = crossfit_ne73(y)
    estimate, mask = moment_ne75(y, prediction, common73)
    ne_placebos76[lag] = estimate
print("Pre-treatment placebos:", ne_placebos76)

unique_stays76, group_index76 = np.unique(groups73, return_inverse=True)
cluster_stats76 = {}
for h in ne_horizons75:
    mask = ne_masks75[h]
    z = ne_z75[mask]
    inverse = group_index76[mask]
    nstays = len(unique_stays76)
    denominator = np.bincount(inverse, weights=z*z, minlength=nstays)
    numerator = np.bincount(
        inverse, weights=z*(ne_future_y75[h]-ne_future_pred_y75[h])[mask],
        minlength=nstays
    )
    dose_numerators = np.column_stack([
        np.bincount(
            inverse, weights=z*(ne_future_d75[j]-ne_future_pred_d75[j])[mask],
            minlength=nstays
        ) for j in range(h)
    ])
    cluster_stats76[h] = denominator, numerator, dose_numerators

rng76 = np.random.default_rng(7601)
bootstrap76 = []
grid76 = np.linspace(0.2, 1.1, 901)
for rep in range(160):
    weights = rng76.multinomial(len(unique_stays76), np.full(len(unique_stays76), 1/len(unique_stays76)))
    estimates, moments = {}, {}
    for h in ne_horizons75:
        denominator, numerator, dose_numerators = cluster_stats76[h]
        total = np.dot(weights, denominator)
        estimates[h] = {
            "beta": float(np.dot(weights, numerator)/total),
            "se": ne_dynamic75[h]["se"]
        }
        moments[h] = weights @ dose_numerators / total
    fit = fit_impulse76(ne_horizons75, estimates, moments, grid76)
    bootstrap76.append((fit["beta"], fit["phi"]))
bootstrap76 = np.asarray(bootstrap76)
print("Cluster-bootstrap beta/phi 2.5%,50%,97.5%:",
      np.quantile(bootstrap76, [0.025, 0.5, 0.975], axis=0))
bootstrap_targets76 = np.array([
    record_schedule_target69(beta, phi, "norepinephrine", "ne_x2")["mean"]
    for beta, phi in bootstrap76
])
print("NE x2 target cluster interval:",
      np.quantile(bootstrap_targets76, [0.025, 0.5, 0.975]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 76: independent MAP channels and causal uncertainty\n"
            f"Arterial/cuff dynamic checks={ne_checks76}; pre-treatment placebos={ne_placebos76}; "
            f"cluster-bootstrap beta/phi quantiles={np.quantile(bootstrap76, [0.025,0.5,0.975], axis=0).tolist()}; "
            f"NE x2 target quantiles={np.quantile(bootstrap_targets76, [0.025,0.5,0.975]).tolist()}. "
            "No weights changed. Python runs: 76; forwards: 10.\n")
```
