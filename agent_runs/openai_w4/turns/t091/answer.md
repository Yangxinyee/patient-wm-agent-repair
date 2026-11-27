We have independent training and tuning features, but the existing dose-innovation decoders transfer only weakly. I’ll test a different approach: a small, analytically fitted scalar calibration whose dose response is constrained by the record estimates. I won’t save weights yet. A useful candidate must restore both response directions while keeping tuning NLL within the allowed margin.

```python
import numpy as np
import pandas as pd
from scipy.linalg import solve

# A separate scalar regression; its coefficients would imply a rank-one
# edit of the final value-bin layer. No model gradients or optimiser.
tilt_bins80 = np.r_[0.0, np.linspace(-2.0, 2.0, 16)]
mu_phi80 = phi74.astype(np.float64).mean(axis=0)
X80 = np.c_[phi74.astype(np.float64) - mu_phi80, np.ones(len(phi74))]
XT80 = np.c_[head_phi_repair.astype(np.float64) - mu_phi80,
             np.ones(len(head_phi_repair))]
P80 = prob74.astype(np.float64)
PT80 = head_prob_repair.astype(np.float64)
y80 = head_train_data_repair["observed_bin"].to_numpy(dtype=int)
yt80 = head_tuning_data_repair["observed_bin"].to_numpy(dtype=int)
et80 = P80 @ tilt_bins80
vt80 = np.maximum(P80 @ (tilt_bins80**2) - et80**2, 1e-5)
target_fact80 = (tilt_bins80[y80] - et80) / vt80
H80 = (X80.T * vt80) @ X80 / len(X80)
g80 = X80.T @ (vt80 * target_fact80) / len(X80)

G80 = np.sum(P80 * (centres74 - mean74[:, None]) *
             tilt_bins80[None, :], axis=1)
causal_beta80 = {"insulin": -2.04924, "norepinephrine": 3.17146}
A_parts80, b_parts80 = [], []
pair80 = {}
for key, obj in one_step74.items():
    pc = np.asarray(obj["prob"], dtype=np.float64)
    xc = np.c_[np.asarray(obj["phi"], dtype=np.float64) - mu_phi80,
               np.ones(len(X80))]
    mc = np.sum(pc * centres74, axis=1)
    gc = np.sum(pc * (centres74 - mc[:, None]) *
                tilt_bins80[None, :], axis=1)
    mask = np.asarray(obj["eligible"], dtype=bool)
    dr = np.asarray(obj["delta_rate"], dtype=np.float64)
    mask &= np.abs(dr) > 1e-8
    scale = np.maximum(np.abs(G80[mask]), 0.5)
    aa = (gc[mask, None] * xc[mask] -
          G80[mask, None] * X80[mask]) / scale[:, None]
    bb = (causal_beta80[key[0]] * dr[mask] -
          np.asarray(obj["delta_mean"])[mask]) / scale
    A_parts80.append(aa)
    b_parts80.append(bb)
    pair80[key] = (xc, pc, mask, dr)
A80 = np.vstack(A_parts80)
b80 = np.concatenate(b_parts80)
HA80 = A80.T @ A80 / len(A80)
ga80 = A80.T @ b80 / len(A80)
ridge_scale80 = np.trace(H80) / H80.shape[0]

def calibrated_prob80(p, x, coef):
    shift = np.clip((x @ coef)[:, None] * tilt_bins80[None, :], -40, 40)
    pp = p * np.exp(shift)
    return pp / pp.sum(axis=1, keepdims=True)

def metrics80(coef):
    p = calibrated_prob80(P80, X80, coef)
    pt = calibrated_prob80(PT80, XT80, coef)
    out = {
        "train_dNLL": float(np.mean(np.log(P80[np.arange(len(y80)), y80]) -
                                   np.log(p[np.arange(len(y80)), y80]))),
        "tuning_dNLL": float(np.mean(np.log(PT80[np.arange(len(yt80)), yt80]) -
                                    np.log(pt[np.arange(len(yt80)), yt80]))),
        "tilt_SD": float(np.std(XT80 @ coef)),
        "coef_norm": float(np.linalg.norm(coef)),
    }
    mb = np.sum(p * centres74, axis=1)
    for key, (xc, pc, mask, dr) in pair80.items():
        pp = calibrated_prob80(pc, xc, coef)
        dm = np.sum(pp * centres74, axis=1) - mb
        out[f"{key[0]}_{key[1]}"] = float(
            np.dot(dm[mask], dr[mask]) / np.dot(dr[mask], dr[mask]))
    return out

calibration_candidates80 = {}
rows80 = []
for alpha in [0.01, 0.1, 1.0, 10.0]:
    for strength in [0.0, 0.1, 1.0, 10.0, 100.0]:
        reg = alpha * ridge_scale80 * np.eye(X80.shape[1])
        coef = solve(H80 + strength * HA80 + reg,
                     g80 + strength * ga80, assume_a="pos")
        calibration_candidates80[(alpha, strength)] = coef
        rows80.append({"alpha": alpha, "strength": strength, **metrics80(coef)})
calibration_table80 = pd.DataFrame(rows80)
print("Original approximate NLL:",
      -np.log(P80[np.arange(len(y80)), y80]).mean(),
      -np.log(PT80[np.arange(len(yt80)), yt80]).mean())
print("Mean derivative per unit tilt by event:")
for event in ["glucose_finger", "map_arterial"]:
    mask = head_train_data_repair.event.eq(event).to_numpy()
    print(event, np.quantile(G80[mask], [0.1, 0.5, 0.9]))
print(calibration_table80.round(5).to_string(index=False))
print("Weights remain unedited:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 80\n"
            "Fit closed-form scalar calibration regressions using factual "
            "bin residuals and paired dose-feature response constraints. "
            "Compared independent tuning NLL and nonlinear one-step responses "
            "over a ridge/constraint grid. No weights saved.\n")
```
