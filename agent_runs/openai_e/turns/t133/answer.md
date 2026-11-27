The full-strength correction perturbs many factual bundles, including bundles without norepinephrine; some perturbations are large. I’ll test a different detector target: recorded dose minus the dose expected from the recorded history, while suppressing activation elsewhere. This approach would be promising if held-out activations track dose innovations and retain balanced doubling/halving sensitivity with less unrelated activation.

```python
import numpy as np
import pandas as pd
from pathlib import Path

def np115(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

ncf115 = len(cf_frame)
rows115 = cf_frame["state_row"].to_numpy(dtype=int)
assert np.array_equal(state_df["state_row"].to_numpy(), np.arange(len(state_df)))

def bundle_target115(key):
    y = np115(Y81[key]).astype(np.float64).reshape(-1)
    if y.size == ncf115:
        return y
    if y.size == 4 * ncf115:
        return y.reshape(ncf115, 4).mean(axis=1)
    raise ValueError((key, y.shape, ncf115))

raw_ne115 = bundle_target115(("norepinephrine", 2.0))
print("NE contrast target range:", np.quantile(raw_ne115, [0, .5, .9, .99, 1]))
print("halving consistency:",
      np.max(np.abs(bundle_target115(("norepinephrine", .5)) + .5 * raw_ne115)))

policy_pred115 = dose_policy["norepinephrine"].predict(dose_covariates(tpanel))
policy_table115 = tpanel[["stay_id", "t", "norepinephrine"]].copy()
policy_table115["expected_ne"] = policy_pred115
keys115 = cf_frame[["stay_id", "hour"]].copy()
keys115["t"] = np.floor(keys115["hour"] + 1e-4).astype(int)
joined115 = keys115.merge(policy_table115, on=["stay_id", "t"], how="left",
                         validate="many_to_one", sort=False)
assert len(joined115) == ncf115
active115 = raw_ne115 > 1e-8
assert joined115.loc[active115, "expected_ne"].notna().all()
print("raw-dose / panel alignment RMS:",
      np.sqrt(np.mean((raw_ne115[active115] -
                       joined115.loc[active115, "norepinephrine"].to_numpy())**2)))

innovation_cf115 = np.zeros(ncf115)
innovation_cf115[active115] = (
    raw_ne115[active115] -
    joined115.loc[active115, "expected_ne"].to_numpy()
)
innovation_all115 = np.zeros(len(state_df))
innovation_all115[rows115] = innovation_cf115
active_all115 = np.zeros(len(state_df), dtype=bool)
active_all115[rows115] = active115

score_unit115 = scores114.mean(axis=1) / ne_metrics103["s2"]
test_b115 = state_df["stay_id"].to_numpy() % 5 == 0
test_c115 = cf_frame["stay_id"].to_numpy() % 5 == 0
valid115 = test_c115 & active115
print("held-out innovation n / mean / SD:",
      valid115.sum(), innovation_cf115[valid115].mean(),
      innovation_cf115[valid115].std())
print("old detector / innovation correlation:",
      np.corrcoef(score_unit115[rows115][valid115],
                  innovation_cf115[valid115])[0, 1])

scale115 = np115(scale95).astype(np.float64)
null115 = np115(null95).astype(np.float64)
Ef115 = np115(E_fact95).astype(np.float64)
Xf115 = (Ef115.reshape(-1, 192) / scale115) @ null115
train_f115 = np.repeat(~test_b115, 4)
active_f115 = np.repeat(active_all115, 4)
target_f115 = np.repeat(innovation_all115, 4)
train_c115 = np.repeat(~test_c115, 4)

Dc115, yc115 = {}, {}
for key in E_edit95:
    difference = np115(E_edit95[key]).astype(np.float64) - Ef115[rows115]
    Dc115[key] = (difference.reshape(-1, 192) / scale115) @ null115
    yc115[key] = np.repeat(bundle_target115(key), 4)

def moment115(X, y, mask):
    xx, yy = X[mask], y[mask]
    return xx.T @ xx / len(xx), xx.T @ yy / len(xx)

G_active115, R_active115 = moment115(
    Xf115, target_f115, train_f115 & active_f115)
G_other115, R_other115 = moment115(
    Xf115, target_f115, train_f115 & ~active_f115)
G_cf115, R_cf115 = {}, {}
for key in Dc115:
    G_cf115[key], R_cf115[key] = moment115(
        Dc115[key], yc115[key] if key[0] == "norepinephrine"
        else np.zeros_like(yc115[key]), train_c115)

fits115, metrics115 = {}, []
for other_weight in [3., 10., 30., 100., 300.]:
    for factual_weight in [.3, 1., 3.]:
        for half_weight in [1., 3.]:
            A = factual_weight * G_active115 + other_weight * G_other115
            rhs = factual_weight * R_active115
            for key in Dc115:
                weight = ((half_weight if key[1] == .5 else 1.)
                          if key[0] == "norepinephrine" else 30.)
                A = A + weight * G_cf115[key]
                rhs = rhs + weight * R_cf115[key]
            ridge = 1e-5 * np.trace(A) / len(A)
            theta = np.linalg.solve(A + ridge * np.eye(len(A)), rhs)
            feature = (null115 @ theta) / scale115
            choice = (other_weight, factual_weight, half_weight)
            fits115[choice] = feature

            pf = Ef115 @ feature
            metric = {"choice": choice}
            for factor in [2., .5]:
                p = (np115(E_edit95[("norepinephrine", factor)]) -
                     Ef115[rows115]) @ feature
                p = p.mean(axis=1)
                y = bundle_target115(("norepinephrine", factor))
                m = test_c115 & active115
                metric["gain_" + str(factor)] = float(p[m] @ y[m] / (y[m] @ y[m]))
            gain = metric["gain_2.0"]
            metric["half_ratio"] = metric["gain_0.5"] / gain if gain else np.nan
            metric["other_RMS_per_gain"] = float(
                np.sqrt(np.mean(pf[test_b115 & ~active_all115]**2)) / abs(gain))
            pin = pf.mean(axis=1)[rows115][valid115] / gain
            metric["innovation_RMSE"] = float(
                np.sqrt(np.mean((pin - innovation_cf115[valid115])**2)))
            metric["innovation_corr"] = float(
                np.corrcoef(pin, innovation_cf115[valid115])[0, 1])
            metrics115.append(metric)

table115 = pd.DataFrame(metrics115)
print("\n=== INNOVATION-TARGETED DETECTORS ===")
print(table115.sort_values("other_RMS_per_gain").head(12).to_string(index=False))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 115: factual innovation-targeted NE detectors\n"
            "Full trial activates non-NE factual bundles and has large hnew perturbations. "
            "Fitted closed-form prelude probes targeting record dose innovations on active "
            "NE bundles and zero elsewhere, with synthetic dose contrasts, insulin guards, "
            "and admission-null constraints. Held-out stay diagnostics printed. "
            "No weights changed. Python runs: 115; forwards: 15.\n")
```