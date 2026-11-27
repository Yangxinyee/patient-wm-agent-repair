The corrected head’s discrepancy is now comparable to four-draw latent variability, which is sufficient for preliminary screening—not a substitute for GPU validation. Before editing, I’ll estimate how drug effects persist across hours. The next-hour estimates alone do not determine the twelve-hour readouts; later-dose-adjusted impulse responses should distinguish rapid decay from accumulation.

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

print("=== PANEL AND NUISANCE METADATA ===")
print("panel index:", pan.index.names, "columns:", list(pan.columns))
print("history features:", list(HX.columns))
for drug, rr in dml_residuals.items():
    print(drug, {
        k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
        for k, v in rr.items()
    })
    mm = dml_models[drug]
    if isinstance(mm, dict):
        print("models:", {k: type(v).__name__ for k, v in mm.items()})
    else:
        print("models:", type(mm).__name__)
print("\nrollout columns:", list(base_roll.columns))
print(base_roll.head(5).to_string(index=False))

print("\n=== MEASUREMENT PAIR CHECKS ===")
for a_event, b_event in [
    ("glucose_finger", "glucose_lab"), ("map_arterial", "map_cuff")
]:
    pair = pan[[a_event, b_event]].dropna()
    dd = pair[a_event] - pair[b_event]
    print(a_event, b_event, "paired n:", len(pair),
          "difference mean/SD:", round(dd.mean(), 4), round(dd.std(), 4),
          "correlation:", round(pair.corr().iloc[0, 1], 5))

panel_stays_repair = pan.index.get_level_values(0).to_numpy()
impulse_tables_repair = {}
persistence_repair = {}
for drug, target in [
    ("insulin", "glucose_finger"), ("norepinephrine", "map_arterial")
]:
    rr = dml_residuals[drug]
    if not all(k in rr for k in ("indices", "dose", "dose_prediction")):
        print("Missing residual fields; skipping impulse diagnostic for", drug)
        continue
    ix = np.asarray(rr["indices"], dtype=int)
    eps = np.asarray(rr["dose"]) - np.asarray(rr["dose_prediction"])
    current_dose = pan[drug].to_numpy(dtype=float)
    current_y = last[target].to_numpy(dtype=float)
    future_doses = [
        pan[drug].groupby(level=0).shift(-j).to_numpy(dtype=float)
        for j in range(6)
    ]
    rows_imp = []
    moment_data = []
    for horizon in range(1, 7):
        future_y = pan[target].groupby(level=0).shift(-horizon).to_numpy(dtype=float)
        ok_imp = (
            np.isfinite(eps) & np.isfinite(future_y[ix])
            & np.isfinite(current_y[ix]) & np.isfinite(current_dose[ix])
        )
        jj = ix[ok_imp]
        ee = eps[ok_imp].astype(float)
        ee -= ee.mean()
        dy = future_y[jj] - current_y[jj]
        dy -= dy.mean()
        dd = current_dose[jj].astype(float)
        dd -= dd.mean()
        denom = ee @ dd
        if abs(denom) < 1e-8:
            continue
        response = float((ee @ dy) / denom)
        influence = ee * (dy-response*dd) / denom
        group_code, groups = pd.factorize(panel_stays_repair[jj])
        cluster_sum = np.bincount(group_code, weights=influence)
        se = float(np.sqrt(
            len(groups)/(len(groups)-1) * np.sum(cluster_sum**2)
        ))
        dose_covariances = []
        for lag in range(horizon):
            fd = future_doses[lag][jj]
            dose_covariances.append(float(
                ee @ (fd-fd.mean()) / denom
            ))
        rows_imp.append((
            horizon, len(jj), response, se, *dose_covariances,
            *([np.nan] * (6-horizon))
        ))
        moment_data.append((horizon, response, se, np.asarray(dose_covariances)))
    tab_imp = pd.DataFrame(rows_imp, columns=[
        "horizon", "n", "closed_loop_response", "cluster_se",
        "dose_cov_0", "dose_cov_1", "dose_cov_2",
        "dose_cov_3", "dose_cov_4", "dose_cov_5"
    ])
    beta = float(dml_results.loc[dml_results.drug.eq(drug), "beta"].iloc[0])
    def persistence_loss(rho):
        return sum(
            ((response-beta*np.dot(rho**np.arange(k-1, -1, -1), dc))
             / max(se, .03))**2
            for k, response, se, dc in moment_data
        )
    rho_fit = minimize_scalar(persistence_loss, bounds=(0, 1.15), method="bounded")
    rho = float(rho_fit.x)
    tab_imp["AR_predicted_response"] = [
        beta*np.dot(rho**np.arange(k-1, -1, -1), dc)
        for k, response, se, dc in moment_data
    ]
    impulse_tables_repair[drug] = tab_imp
    persistence_repair[drug] = {
        "beta": beta, "rho": rho, "weighted_loss": float(rho_fit.fun),
        "n_moments": len(moment_data)
    }
    print("\n=== IMPULSE DIAGNOSTIC:", drug, "===")
    print(tab_imp.round(5).to_string(index=False))
    print("Provisional persistence fit:", persistence_repair[drug])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 19\n"
        "Corrected head/register reconstruction has residual errors comparable "
        "to four-draw prior variability. Began longitudinal causal diagnostics: "
        "dose innovations instrument later outcomes, explicitly accounting "
        "for covariance with subsequent recorded doses. AR persistence fits "
        "are provisional because later observation selection and nuisance "
        "misspecification can bias these moments: "
        + repr(persistence_repair) + ". No weights edited.\n"
    )
```