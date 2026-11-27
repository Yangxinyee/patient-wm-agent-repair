The strongest readout fits damage factual predictions and are unsuitable. A more regularised fit gives a modest sensitivity gain and improves the reconstructed held-out loss, but that is only a proxy for the real score. I’ll install that reversible candidate and test it with the GPU factual tool. A loss increase above 0.005 would reject it; a pass would justify a rollout test.

```python
import numpy as np
import pandas as pd
import json
from scipy.linalg import solve

repair_head_fw = 0.1
repair_head_alpha = 100.0
repair_head_mean = repair_fH.mean(0)
repair_head_sd = np.maximum(repair_fH.std(0), 0.02)
Xf = (repair_fH - repair_head_mean) / repair_head_sd
Xc = repair_DH / repair_head_sd
A = repair_head_fw * (Xf.T @ (repair_fvar[:, None] * Xf))
A += Xc.T @ (repair_W[:, None] * Xc)
A += repair_head_alpha * np.eye(192)
rhs = Xc.T @ (repair_W * repair_T)
repair_head_coef = solve(A, rhs, assume_a="pos") / repair_head_sd
repair_head_intercept = -repair_head_mean @ repair_head_coef

repair_head_W0 = load_original("attr_heads.value_bin.2.weight")
repair_head_b0 = load_original("attr_heads.value_bin.2.bias")
repair_head_dW = np.outer(repair_axis, repair_head_coef).astype(np.float32)
repair_head_db = (repair_axis * repair_head_intercept).astype(np.float32)
save_edit("attr_heads.value_bin.2.weight", repair_head_W0 + repair_head_dW)
save_edit("attr_heads.value_bin.2.bias", repair_head_b0 + repair_head_db)

tau_f = repair_fH @ repair_head_coef + repair_head_intercept
p_f = repair_tilt(repair_fP, tau_f)
idx = np.arange(len(head_meta))
dnll = np.log(np.maximum(repair_fP[idx, repair_fbins], 1e-30)) - np.log(
    np.maximum(p_f[idx, repair_fbins], 1e-30))
print("Training reconstruction dNLL:", float(dnll.mean()))
print("Factual tilt RMS:", float(np.sqrt(np.mean(tau_f**2))))

tau_b = repair_base_H @ repair_head_coef + repair_head_intercept
mu_b = (repair_tilt(repair_base_P, tau_b) * repair_C17).sum(1)
for key, cf in repair_readout_cf.items():
    tau_c = cf["hidden"] @ repair_head_coef + repair_head_intercept
    mu_c = (repair_tilt(cf["probs"], tau_c) * repair_C17).sum(1)
    primary = "glucose_finger" if key[0] == "insulin" else "map_arterial"
    m = cf["active"] & (events == primary)
    dd = cf["delta_rate"][m]
    print(key, "local slope:", float(dd @ (mu_c[m] - mu_b[m]) / (dd @ dd)))

repair_tuning_ids = pd.read_csv("/data/tuning.csv", usecols=["stay_id"]).stay_id.unique().tolist()
repair_validation_ids = repair_tuning_ids[:100]
print("FORWARD STAYS:", json.dumps(repair_validation_ids))
print("ACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 80: reversible value-head contrast candidate\n"
            "- Installed a rank-one ordinal-bin tilt in attr_heads.value_bin.2.weight and bias only.\n"
            "- Decoder fitted to local dose contrasts with factual_weight=0.1, alpha=100; no model gradients used.\n"
            "- Highly aggressive contrast fits failed factual preservation; selected a regularised candidate.\n"
            "- This is not accepted until GPU held-out factual and paired rollout checks pass.\n"
            "- Forward passes used: 11.\n")
```