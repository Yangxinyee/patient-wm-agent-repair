The recovered inputs let us fit a small conditional calibration rather than another global bin bias. I’ll fit it on one training-stay partition and test it on the other partition and tuning records, without saving weights. Lower held-out loss without erasing norepinephrine sensitivity would create useful repair budget.

```python
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

def mixture_nll190(logits, target):
    probs = logits.softmax(-1)
    target = torch.as_tensor(target, dtype=torch.long)
    pobs = probs.gather(
        -1, target[None, :, None].expand(probs.shape[0], -1, 1)
    ).squeeze(-1)
    return -pobs.reshape(-1, 4, len(target)).mean(1).clamp_min(1e-30).log().mean(0).numpy()

def scalar_nll190(mc, base_logits, coef, mask):
    mask = np.asarray(mask, dtype=bool)
    xx = mc["inputs"][:, mask, :]
    cc = torch.as_tensor(coef, dtype=torch.float32)
    tau = xx @ cc[:-1] + cc[-1]
    return mixture_nll190(
        base_logits[:, mask, :] + tau[:, :, None] * wm_v151,
        mc["target"][mask]
    )

cal190_base_train = head177_eval(wm_train145, head177_weights)
cal190_base_tune = head177_eval(wm_tune145, head177_weights)
cal190_fitmask = wm_split151 & map172_MAP_train
cal190_cvmask = (~wm_split151) & map172_MAP_train

xx190 = wm_train145["inputs"][:, cal190_fitmask, :].numpy().astype(np.float64)
FF190 = np.concatenate(
    [xx190, np.ones((*xx190.shape[:2], 1))], axis=-1
)
pp190 = cal190_base_train["probs"][:, cal190_fitmask, :]
vv190 = wm_v151
mu190 = (pp190 * vv190).sum(-1)
var190 = (pp190 * vv190.square()).sum(-1) - mu190.square()
yy190 = wm_train145["target"][cal190_fitmask].long()
pobs190 = pp190.gather(
    -1, yy190[None, :, None].expand(pp190.shape[0], -1, 1)
).squeeze(-1)
resp190 = pobs190.reshape(-1, 4, len(yy190))
resp190 = (resp190 / resp190.sum(1, keepdim=True).clamp_min(1e-30)).reshape_as(pobs190)
res190 = vv190[yy190][None, :] - mu190

flat190 = FF190.reshape(-1, 193)
curv190 = (4 * resp190 * var190).numpy().reshape(-1).astype(np.float64)
score190 = (4 * resp190 * res190).numpy().reshape(-1).astype(np.float64)
cal190_K = (flat190.T @ (flat190 * curv190[:, None])) / len(flat190)
cal190_b = (flat190.T @ score190) / len(flat190)
cal190_R = np.diag(np.r_[map164_scale**2, 1.0])

orig190_train = wm_train145["nll"].mean(0).numpy()
orig190_tune = wm_tune145["nll"].mean(0).numpy()
print("CURRENT LOCAL TOTAL dNLL:",
      "heldout", np.mean((cal190_base_train["nll"] - orig190_train)[~wm_split151]),
      "tuning", np.mean(cal190_base_tune["nll"] - orig190_tune))
print("LOCAL CONTRAST OBJECT KEYS:",
      {k: list(v.keys()) for k, v in wm_cf146.items()})

cal190_candidates = {}
cal190_rows = []
for ridge190 in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    for protection190 in [0.0, 10.0]:
        delta190 = np.linalg.solve(
            cal190_K + ridge190 * cal190_R + protection190 * map164_C_ne,
            cal190_b
        )
        cal190_candidates[(ridge190, protection190)] = delta190
        trn190 = cal190_base_train["nll"].copy()
        tun190 = cal190_base_tune["nll"].copy()
        trn190[map172_MAP_train] = scalar_nll190(
            wm_train145, cal190_base_train["logits"], delta190, map172_MAP_train
        )
        tun190[map172_MAP_tune] = scalar_nll190(
            wm_tune145, cal190_base_tune["logits"], delta190, map172_MAP_tune
        )
        cal190_rows.append({
            "ridge": ridge190, "protect": protection190,
            "fit_MAP_gain": np.mean((trn190-cal190_base_train["nll"])[cal190_fitmask]),
            "cv_MAP_gain": np.mean((trn190-cal190_base_train["nll"])[cal190_cvmask]),
            "tune_MAP_gain": np.mean((tun190-cal190_base_tune["nll"])[map172_MAP_tune]),
            "cv_total_dNLL": np.mean((trn190-orig190_train)[~wm_split151]),
            "tune_total_dNLL": np.mean(tun190-orig190_tune),
            "NE_contrast_rms": np.sqrt(max(0, delta190 @ map164_C_ne @ delta190)),
            "norm": np.linalg.norm(delta190)
        })
cal190_table = pd.DataFrame(cal190_rows)
print(cal190_table.to_string(index=False))
cal190_table.to_csv("/work/map_calibration_frontier.csv", index=False)
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 190: conditional MAP calibration experiment\n"
            "- Fitted ridge-regularised scalar calibration to factual fine-bin residuals on training-stay partition only.\n"
            "- Used conditional prior-draw responsibilities and positive Fisher curvature; no model backpropagation or optimiser.\n"
            "- Evaluated held-out training partition and tuning records; no new weights saved.\n"
            + cal190_table.to_string(index=False) + "\n")
```
