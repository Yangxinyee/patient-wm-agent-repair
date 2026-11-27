The diagnostic stopped on a metadata mismatch, before fitting anything; no weights changed. I’ll derive the outcome masks directly from event names and rerun. Held-out decoding accuracy—not the training fit—will determine whether these features are useful.

```python
import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import r2_score

decode_meta_sets = {
    "first": head_meta_train,
    "extra": head_meta_extra,
    "tuning": head_meta_tune,
}
decode_results = {}
decode_models = {}

for j, drug in enumerate(["insulin", "norepinephrine"]):
    outcomes = (["glucose_finger", "glucose_lab"] if j == 0
                else ["map_arterial", "map_cuff"])
    masks = {
        s: np.asarray(decode_dose_sets[s]["valid"], bool)
           & m.event.isin(outcomes).to_numpy()
        for s, m in decode_meta_sets.items()
    }
    for fname in decode_feature_sets["first"]:
        A = decode_feature_sets["first"][fname][masks["first"]]
        location = A.mean(0)
        scale = np.maximum(A.std(0), 0.05)
        A = (A - location) / scale
        B = (decode_feature_sets["extra"][fname][masks["extra"]] - location) / scale
        C = (decode_feature_sets["tuning"][fname][masks["tuning"]] - location) / scale
        ev, U = eigh(A.T @ A)
        ev = np.maximum(ev, 0.0)
        for target_name in ["residual", "doses"]:
            yy = np.asarray(decode_dose_sets["first"][target_name])[masks["first"], j]
            yb = np.asarray(decode_dose_sets["extra"][target_name])[masks["extra"], j]
            yc = np.asarray(decode_dose_sets["tuning"][target_name])[masks["tuning"], j]
            intercept = yy.mean()
            cross = U.T @ (A.T @ (yy - intercept))
            trials = []
            for alpha in [0.1, 1, 10, 100, 1000, 10000]:
                ww = U @ (cross / (ev + alpha))
                pb, pc = intercept + B @ ww, intercept + C @ ww
                trials.append((r2_score(yb, pb), alpha, r2_score(yc, pc),
                               float(pc.std()), ww))
            best = max(trials, key=lambda z: z[0])
            key = (drug, fname, target_name)
            decode_results[key] = {
                "alpha": best[1], "extra_r2": round(best[0], 4),
                "tuning_r2": round(best[2], 4),
                "prediction_sd": round(best[3], 4),
                "target_sd": round(float(yc.std()), 4)
            }
            decode_models[key] = (
                best[4] / scale,
                intercept - location @ (best[4] / scale)
            )
            print(key, decode_results[key])
```
