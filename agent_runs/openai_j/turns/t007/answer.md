The original factual value NLL is 1.63164 on these 100 tuning stays, so an edit must preserve it closely. I’ll fit small latent dynamical models to estimate how drug effects persist, accounting for sensor noise. Agreement with the residualised dose estimates would support using these models to set rollout targets; disagreement would call for further investigation.

```python
from scipy.optimize import minimize

print("=== FORWARD FILE FORMATS ===")
dose_file = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
factual_file = pd.read_csv("/work/fwd_2/factual_original.csv")
print("Dose:", dose_file.shape, dose_file.columns.tolist())
print(dose_file.head(4).to_string(index=False))
print("Factual:", factual_file.shape, factual_file.columns.tolist())
print(factual_file.head(3).to_string(index=False))

# A small statistical model, not the neural world model:
# latent physiology x, slowly varying baseline b, and two noisy sensors.
# x[t+1] = rho*x[t] + (1-rho)*b[t] + beta*recorded_rate[t] + time drift.
def kalman_nll(theta, readings, rates, return_filter=False):
    rho, beta = theta[:2]
    q, qb, s1, s2 = np.exp(theta[2:6])
    mux, mub = theta[6:8]
    sx, sb = np.exp(theta[8:10])
    corr = np.tanh(theta[10])
    offset, d0, d1 = theta[11:14]
    n, steps, _ = readings.shape
    x = np.full(n, mux)
    b = np.full(n, mub)
    p00 = np.full(n, sx*sx)
    p01 = np.full(n, corr*sx*sb)
    p11 = np.full(n, sb*sb)
    nll = 0.0
    filtered = []
    nobs = np.isfinite(readings).sum()
    for t in range(steps):
        for sensor, sigma in enumerate([s1, s2]):
            yy = readings[:, t, sensor]
            valid = np.isfinite(yy)
            error = np.where(valid, yy - x - (offset if sensor == 1 else 0), 0)
            variance = p00 + sigma*sigma
            nll += 0.5*np.sum(
                np.where(valid, np.log(2*np.pi*variance) + error*error/variance, 0)
            )
            k0 = np.where(valid, p00/variance, 0)
            k1 = np.where(valid, p01/variance, 0)
            old00, old01 = p00, p01
            x = x + k0*error
            b = b + k1*error
            p00 = old00 - k0*old00
            p01 = old01 - k0*old01
            p11 = p11 - k1*old01
        if return_filter:
            filtered.append(x.copy())
        if t < steps - 1:
            x = rho*x + (1-rho)*b + beta*rates[:, t] + d0 + d1*(t-12)/12
            old00, old01, old11 = p00, p01, p11
            p00 = rho*rho*old00 + 2*rho*(1-rho)*old01 + (1-rho)**2*old11 + q*q
            p01 = rho*old01 + (1-rho)*old11
            p11 = old11 + qb*qb
    if return_filter:
        return nll/nobs, np.stack(filtered, axis=1)
    return nll/nobs

fit_indices = np.arange(0, len(train_ids), 6)  # 5,000 stays, chosen without outcomes.
kalman_fits = {}
fit_rows = []
for drug_j, sensors in [(0, [0, 1]), (1, [2, 3])]:
    drug = DRUGS[drug_j]
    readings = Y[fit_indices][:, :, sensors]
    rates = A[fit_indices, :, drug_j]
    initial_mean = float(np.nanmean(readings[:, :2, 0]))
    initial_noise = [8, 4] if drug_j == 0 else [3, 6]
    theta0 = np.array([
        .85, -2 if drug_j == 0 else 3,
        np.log(5 if drug_j == 0 else 2), np.log(.2),
        np.log(initial_noise[0]), np.log(initial_noise[1]),
        initial_mean, initial_mean,
        np.log(35 if drug_j == 0 else 15),
        np.log(35 if drug_j == 0 else 15),
        np.arctanh(.8), 0 if drug_j == 0 else 3, 0, 0
    ])
    bounds = [
        (.05, .995), (-20, 20), (-4, 4), (-8, 3),
        (-2, 3.5), (-2, 3.5),
        (initial_mean-100, initial_mean+100),
        (initial_mean-100, initial_mean+100),
        (0, 5), (0, 5), (-2.5, 2.5), (-20, 20),
        (-20, 20), (-20, 20)
    ]
    t0 = time.time()
    fit = minimize(kalman_nll, theta0, args=(readings, rates),
                   method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 180, "ftol": 1e-9, "maxls": 30})
    kalman_fits[drug] = fit
    theta = fit.x
    row = {
        "drug": drug, "rho": theta[0], "beta": theta[1],
        "process_sd": np.exp(theta[2]), "baseline_walk_sd": np.exp(theta[3]),
        "sensor1_sd": np.exp(theta[4]), "sensor2_sd": np.exp(theta[5]),
        "sensor2_offset": theta[11], "nll": fit.fun,
        "success": fit.success, "seconds": time.time()-t0
    }
    fit_rows.append(row)
print("\n=== LATENT DYNAMICAL FITS ===")
print(pd.DataFrame(fit_rows).round(5).to_string(index=False))

TA = tuning_panel[DRUGS].to_numpy().reshape(len(tuning_ids), 26, 2)
surrogate_rows = []
for j, drug in enumerate(DRUGS):
    rho, beta = kalman_fits[drug].x[:2]
    for factor in [0, .5, 1.5, 2, 3]:
        change = np.zeros(len(tuning_ids))
        reached = np.zeros(len(tuning_ids), dtype=bool)
        values = []
        first100 = []
        for hour in range(25):
            idx = hour + 1
            if hour >= 12:
                reached |= TA[:, idx, j] > 0
                change = rho*change + beta*(factor-1)*TA[:, idx, j]
                if hour < 24:
                    values.extend(change[reached].tolist())
                    first100.extend(change[:100][reached[:100]].tolist())
        surrogate_rows.append({
            "drug": drug, "factor": factor,
            "mean_shift_all400": np.mean(values),
            "mean_shift_first100": np.mean(first100),
        })
print("\n=== PROVISIONAL FIXED-DOSE ROLLOUT TARGETS ===")
print(pd.DataFrame(surrogate_rows).round(4).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Factual baseline and latent dynamics\n")
    f.write("- Original factual first 100 tuning: value NLL 1.63164, 4,751 readings.\n")
    f.write(pd.DataFrame(fit_rows).to_string(index=False) + "\n")
    f.write(pd.DataFrame(surrogate_rows).to_string(index=False) + "\n")
    f.write("- Targets above are provisional statistical estimates, not simulator ground truth.\n")
    f.write("- Python runs: 5. Forward passes: 2. No edits yet.\n")
```