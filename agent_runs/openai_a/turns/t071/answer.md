The threshold estimates consistently put insulin’s one-hour effect near −2. I’ll now compare dose decoding from the state, the raw head input, and the hidden head features, using separate stays for validation. Recoverable dose innovations would identify a plausible edit location; failure across these features would push the repair earlier in the action pathway.

```python
import numpy as np
import torch
from scipy.linalg import eigh
from sklearn.metrics import r2_score

def repair_feature_sets(meta, sf):
    parts = cpu_head_parts(meta, sf)
    zmean = parts["prior"][:, :32]
    q0 = ln(parts["base"] + lin(zmean, "z_head")
            + W["type_emb.weight"][1], "query_norm")
    lz = torch.tensor(meta.last_z.to_numpy(), dtype=torch.float32)[:, None]
    add0 = parts["additive"] + lz * (
        W["last_numeric.weight"][:, 6] - W["last_numeric.weight"][:, 2])
    inp = q0 + add0
    hid = torch.nn.functional.gelu(lin(inp, "attr_heads.value_bin.0"))
    xx = np.asarray(sf["x"])[meta.state_row.to_numpy(dtype=int)]
    return {
        "state_out": xx[:, :192].astype(np.float64),
        "head_view": q0.numpy().astype(np.float64),
        "head_input": inp.numpy().astype(np.float64),
        "head_hidden": hid.numpy().astype(np.float64),
    }

decode_feature_sets = {
    "first": repair_feature_sets(head_meta_train, state_files[4]),
    "extra": repair_feature_sets(head_meta_extra, state_files[5]),
    "tuning": repair_feature_sets(head_meta_tune, state_files[3]),
}
decode_dose_sets = {
    "first": head_dose_data(head_meta_train, P, train_ids,
                            policy_predictions_panel),
    "extra": head_dose_data(head_meta_extra, P, train_ids,
                            policy_predictions_panel),
    "tuning": head_dose_data(head_meta_tune, TP, tuning_ids,
                             tpolicy_predictions_panel),
}

print("DOSE DATA KEYS", decode_dose_sets["first"].keys())
print("GROUP KEYS", decode_dose_sets["first"]["groups"].keys())

decode_results = {}
decode_models = {}
for j, drug in enumerate(["insulin", "norepinephrine"]):
    for fname in decode_feature_sets["first"]:
        masks = {
            s: np.asarray(d["valid"], bool) & np.asarray(d["groups"][drug], bool)
            for s, d in decode_dose_sets.items()
        }
        A = decode_feature_sets["first"][fname][masks["first"]]
        location = A.mean(0)
        scale = np.maximum(A.std(0), 0.05)
        A = (A - location) / scale
        B = (decode_feature_sets["extra"][fname][masks["extra"]] - location) / scale
        C = (decode_feature_sets["tuning"][fname][masks["tuning"]] - location) / scale
        eigenvalues, eigenvectors = eigh(A.T @ A)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        for target_name in ["residual", "doses"]:
            yy = np.asarray(decode_dose_sets["first"][target_name])[masks["first"], j]
            yb = np.asarray(decode_dose_sets["extra"][target_name])[masks["extra"], j]
            yc = np.asarray(decode_dose_sets["tuning"][target_name])[masks["tuning"], j]
            intercept = yy.mean()
            cross = eigenvectors.T @ (A.T @ (yy - intercept))
            trials = []
            for alpha in [0.1, 1, 10, 100, 1000, 10000]:
                ww = eigenvectors @ (cross / (eigenvalues + alpha))
                pb, pc = intercept + B @ ww, intercept + C @ ww
                trials.append((r2_score(yb, pb), alpha, r2_score(yc, pc),
                               float(pc.std()), ww))
            best = max(trials, key=lambda z: z[0])
            key = (drug, fname, target_name)
            decode_results[key] = {
                "alpha": best[1], "extra_r2": best[0], "tuning_r2": best[2],
                "tuning_prediction_sd": best[3], "tuning_target_sd": float(yc.std())
            }
            decode_models[key] = (best[4] / scale,
                                  intercept - location @ (best[4] / scale))
            print(key, decode_results[key])
```
