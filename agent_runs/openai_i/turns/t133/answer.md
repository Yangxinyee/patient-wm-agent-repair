The action vectors retain dose magnitude much better than the posterior hidden features: held-stay rate-decoding R² is about 0.95 for both drugs. I’ll now test edited doses, including zero and triple dose. Accurate shifts with little cross-drug leakage would support using this representation for a correction.

```python
import numpy as np
import torch

print("=== ACTION RECONSTRUCTION CHECK ===")
print("first case:", repr(drug_cases110[0])[:1000])
action_probe123 = []
for action_case123 in drug_cases110[:12]:
    action_atoms123 = [atom84(r) for r in action_case123[1]
                       if r["event"] in ("insulin", "norepinephrine")]
    action_probe123.append(pool109(action_atoms123, "encode_action", "zero"))
action_probe123 = torch.stack(action_probe123).detach().cpu().numpy()
action_err123 = np.sqrt(np.mean((action_probe123-action_U122[:12])**2))
print("action-vector RMS disagreement:", action_err123)
assert action_err123 < 1e-5, "Resolve action reconstruction before calibration."

print("\n=== SYNTHETIC DOSE REPRESENTATIONS ===")
action_edit_U123 = {}
action_edit_Y123 = {}
for action_drug123 in ("insulin", "norepinephrine"):
    action_col123 = ("insulin", "norepinephrine").index(action_drug123)
    for action_factor123 in (0.0, 0.5, 1.5, 2.0, 3.0):
        action_vectors123 = []
        for action_case123 in drug_cases110:
            action_atoms123 = []
            for action_row123 in action_case123[1]:
                if action_row123["event"] not in ("insulin", "norepinephrine"):
                    continue
                action_r123 = dict(action_row123)
                if action_r123["event"] == action_drug123:
                    action_r123["rate"] = float(action_r123["rate"]) * action_factor123
                    action_r123["dose"] = float(action_r123["dose"]) * action_factor123
                action_atoms123.append(atom84(action_r123))
            action_vectors123.append(pool109(action_atoms123, "encode_action", "zero"))
        action_key123 = (action_drug123, action_factor123)
        action_edit_U123[action_key123] = torch.stack(action_vectors123).detach().cpu().numpy()
        action_edit_Y123[action_key123] = action_Y122.copy()
        action_edit_Y123[action_key123][:, action_col123] *= action_factor123
print("constructed", len(action_edit_U123), "perturbations")

action_augU123 = np.concatenate([action_U122] + list(action_edit_U123.values()))
action_augY123 = np.concatenate([action_Y122] + list(action_edit_Y123.values()))
action_augtr123 = np.tile(action_tr122, 1+len(action_edit_U123))
action_augX123 = np.column_stack(((action_augU123-action_mu122)/action_sd122,
                                 np.ones(len(action_augU123))))
action_augdec123 = {}
print("\n=== HELD-STAY DOSE-SHIFT CALIBRATION ===")
for action_lam123 in (0.001, 0.01, 0.1, 1.0):
    action_pen123 = np.eye(action_augX123.shape[1])*action_lam123
    action_pen123[-1, -1] = 1e-8
    action_B123 = np.linalg.solve(
        action_augX123[action_augtr123].T @ action_augX123[action_augtr123]+action_pen123,
        action_augX123[action_augtr123].T @ action_augY123[action_augtr123])
    action_augdec123[action_lam123] = action_B123
    action_basepred123 = action_X122 @ action_B123
    action_metrics123 = []
    for action_drug123 in ("insulin", "norepinephrine"):
        action_col123 = ("insulin", "norepinephrine").index(action_drug123)
        action_ds123, action_ts123, action_cross123 = [], [], []
        for action_key123, action_U123 in action_edit_U123.items():
            if action_key123[0] != action_drug123:
                continue
            action_X123 = np.column_stack(((action_U123-action_mu122)/action_sd122,
                                           np.ones(len(action_U123))))
            action_dp123 = action_X123 @ action_B123-action_basepred123
            action_dt123 = action_edit_Y123[action_key123]-action_Y122
            action_mask123 = action_te122 & (action_Y122[:,action_col123]>0)
            action_ds123.extend(action_dp123[action_mask123,action_col123])
            action_ts123.extend(action_dt123[action_mask123,action_col123])
            action_cross123.extend(action_dp123[action_mask123,1-action_col123])
        action_ds123, action_ts123 = np.asarray(action_ds123), np.asarray(action_ts123)
        action_metrics123.extend([
            float(action_ds123@action_ts123/(action_ts123@action_ts123)),
            float(np.sqrt(np.mean((action_ds123-action_ts123)**2))),
            float(np.sqrt(np.mean(np.asarray(action_cross123)**2)))])
    print("ridge", action_lam123,
          "ins slope/RMSE/cross; NE slope/RMSE/cross:", np.round(action_metrics123,4))

print("\n=== LATENT-DIRECTION OBJECTS ===")
for action_name123, action_obj123 in list(globals().items()):
    if action_name123.endswith("104") or action_name123 in ("vg", "vm", "v_g", "v_m"):
        if hasattr(action_obj123, "shape"):
            print(action_name123, tuple(action_obj123.shape),
                  "norm", round(float(np.linalg.norm(np.asarray(action_obj123))),5))
        elif isinstance(action_obj123, (str, int, float, dict)):
            print(action_name123, repr(action_obj123)[:220])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 123 — action representation calibration\n"
            "- Factual action-vector rate decoding passed held-stay tests (R2 approximately 0.95).\n"
            "- Testing rate edits 0, .5, 1.5, 2 and 3 on held training stays; synthetic targets calibrate encoding only.\n"
            "- No weights changed.\n")
```