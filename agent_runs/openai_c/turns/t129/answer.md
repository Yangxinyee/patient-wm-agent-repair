The rollout files contain hour-level shifts, so I can separate response strength from accumulation. I’ll fit a simple impulse-response approximation and check factual residuals against earlier readings. Low rollout persistence plus underused reading history would support a register-path edit; otherwise I’ll return to the action pathway.

```python
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from pathlib import Path

dose_lookup92 = {}
for sid92, part92 in panel_tuning.groupby(level=0, sort=False):
    p92 = part92.droplevel(0)
    dose_lookup92[int(sid92)] = {
        drug92: p92[drug92].to_dict()
        for drug92 in ["insulin", "norepinephrine"]
    }

def fit_rollout_dynamics92(path):
    r92 = pd.read_csv(path)
    answer92 = {}
    for drug92, edits92 in [
        ("insulin", ["insulin_x2", "insulin_half"]),
        ("norepinephrine", ["ne_x2", "ne_half"])
    ]:
        for subset92 in [edits92, [edits92[0]], [edits92[1]]]:
            q92 = r92[r92["reached"].astype(bool) &
                      r92["edit"].isin(subset92)].copy()
            if len(q92) == 0:
                continue
            y92 = q92["shift"].to_numpy(float)
            kernels92 = []
            for row92 in q92.itertuples():
                h92 = int(row92.hour)
                times92 = np.arange(12, h92)
                rates92 = np.array([
                    dose_lookup92[int(row92.stay_id)][drug92].get(int(t92), 0.0)
                    for t92 in times92
                ])
                factor92 = 1.0 if row92.edit.endswith("x2") else -0.5
                kernels92.append((h92 - 1 - times92, rates92 * factor92))
            def solve92(rho92):
                x92 = np.array([np.dot(rates92, rho92 ** ages92)
                                for ages92, rates92 in kernels92])
                beta92 = np.dot(x92, y92) / max(np.dot(x92, x92), 1e-12)
                mse92 = np.mean((y92 - beta92 * x92) ** 2)
                return mse92, beta92
            opt92 = minimize_scalar(lambda rr92: solve92(rr92)[0],
                                    bounds=(0.0, 1.04), method="bounded")
            mse92, beta92 = solve92(opt92.x)
            rho_record92 = 0.8412020161353004 if drug92 == "insulin" else 0.673
            mse_fixed92, beta_fixed92 = solve92(rho_record92)
            answer92["+".join(subset92)] = {
                "n": len(q92), "beta": round(float(beta92), 4),
                "rho": round(float(opt92.x), 4),
                "rmse": round(float(np.sqrt(mse92)), 3),
                "beta_at_record_rho": round(float(beta_fixed92), 4),
                "rmse_at_record_rho": round(float(np.sqrt(mse_fixed92)), 3)
            }
    return answer92

rollout_dynamics92 = {
    "original": fit_rollout_dynamics92("/work/fwd_34/dose_edit_original.csv"),
    "numeric_gain4": fit_rollout_dynamics92("/work/fwd_10/dose_edit_edited.csv")
}
print("ROLLOUT DYNAMICS")
print(rollout_dynamics92)

print("\nFACTUAL RESIDUALS VERSUS LAST SAME-SENSOR READING")
f92 = pd.read_csv("/work/fwd_9/factual_edited.csv")
f92 = f92.sort_values(["stay_id", "event", "hour"])
f92["last_observed"] = f92.groupby(["stay_id", "event"])["observed"].shift()
for ev92, part92 in f92.dropna(subset=["last_observed"]).groupby("event"):
    y92 = (part92["observed"] - part92["predicted_mean"]).to_numpy()
    x92 = (part92["last_observed"] - part92["predicted_mean"]).to_numpy()
    xc92, yc92 = x92 - x92.mean(), y92 - y92.mean()
    slope92 = np.dot(xc92, yc92) / np.dot(xc92, xc92)
    print(ev92, {"n": len(part92), "residual_mean": round(y92.mean(), 3),
                "history_correction_slope": round(slope92, 4),
                "residual_sd": round(y92.std(), 3)})

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 92: effective rollout dynamics\n"
            + str(rollout_dynamics92) + "\n"
            "- These are descriptive fits to model rollouts, not new causal estimates.\n"
            "- Numeric gain 4 remains installed. Forward passes used: 34; 16 remain.\n")
```