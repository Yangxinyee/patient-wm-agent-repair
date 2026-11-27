Feature extraction now works, although its means still differ from the GPU’s—especially for glucose. I’ll use it only to design candidates, not to certify factual fit. Next I’ll estimate each dosing policy using other stays; a reliably decoded dose residual could support an edit with little effect on factual records.

```python
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from pathlib import Path

torch.set_num_threads(8)
t0 = time.time()
measure_cols = [event_col[e] for e in
                ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff")]
drug_cols = [event_col["insulin"], event_col["norepinephrine"]]

def make_policy_data(panel):
    readings = panel[:, :, measure_cols]
    n, nt, nm = readings.shape
    indices = np.where(np.isfinite(readings),
                       np.arange(nt)[None, :, None], -1)
    indices = np.maximum.accumulate(indices, axis=1)
    carry = np.take_along_axis(readings, np.maximum(indices, 0), axis=1)
    carry[indices < 0] = np.nan
    drugs = np.nan_to_num(panel[:, :, drug_cols], nan=0.0)
    pieces = [readings[:, 1:25], carry[:, 1:25],
              np.isfinite(readings[:, 1:25]).astype(np.float32)]
    for lag in (1, 2, 3):
        padded = np.concatenate([
            np.full((n, lag, nm), np.nan, dtype=np.float32), carry
        ], axis=1)
        pieces.append(padded[:, 1:25])
    for lag in (1, 2):
        padded = np.concatenate([
            np.zeros((n, lag, 2), dtype=np.float32), drugs
        ], axis=1)
        pieces.append(padded[:, 1:25])
    pieces.append(np.broadcast_to(carry[:, :1], (n, 24, nm)))
    pieces.append(np.broadcast_to(
        np.arange(24, dtype=np.float32)[None, :, None], (n, 24, 1)))
    x = np.concatenate(pieces, axis=2).reshape(n * 24, -1).astype(np.float32)
    a = drugs[:, 1:25].reshape(n * 24, 2).astype(np.float32)
    return x, a, carry

policy_X, policy_A, policy_carry = make_policy_data(P)
tpolicy_X, tpolicy_A, tpolicy_carry = make_policy_data(TP)
policy_fold = np.repeat(np.arange(len(P)) % 2, 24)
policy_predictions = np.empty_like(policy_A)
tpolicy_predictions = np.zeros_like(tpolicy_A)
dose_policy_models = {}
rng_policy = np.random.default_rng(4107)

with threadpool_limits(limits=8):
    for drug_index, drug in enumerate(("insulin", "norepinephrine")):
        models = []
        for fold in (0, 1):
            train_rows = np.flatnonzero(policy_fold != fold)
            if len(train_rows) > 180000:
                train_rows = rng_policy.choice(train_rows, 180000, replace=False)
            test_rows = np.flatnonzero(policy_fold == fold)
            model = HistGradientBoostingRegressor(
                max_iter=100, max_leaf_nodes=15, max_bins=64,
                learning_rate=0.08, l2_regularization=2.0,
                min_samples_leaf=100, early_stopping=False,
                random_state=4107 + fold)
            model.fit(policy_X[train_rows], policy_A[train_rows, drug_index])
            policy_predictions[test_rows, drug_index] = model.predict(
                policy_X[test_rows]).astype(np.float32)
            tpolicy_predictions[:, drug_index] += (
                0.5 * model.predict(tpolicy_X)).astype(np.float32)
            models.append(model)
        dose_policy_models[drug] = models
        residual = policy_A[:, drug_index] - policy_predictions[:, drug_index]
        tresidual = tpolicy_A[:, drug_index] - tpolicy_predictions[:, drug_index]
        print(drug, "held-out-stay policy R2",
              round(r2_score(policy_A[:, drug_index],
                             policy_predictions[:, drug_index]), 4),
              "train residual SD", round(float(residual.std()), 4),
              "tuning residual SD", round(float(tresidual.std()), 4),
              "seconds", round(time.time() - t0, 1))

policy_predictions_panel = policy_predictions.reshape(len(P), 24, 2)
tpolicy_predictions_panel = tpolicy_predictions.reshape(len(TP), 24, 2)
print("Policy feature count:", policy_X.shape[1])
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Cross-stay dose-policy fitting; Python 51, forward 4\n"
            "- Fitted small boosted-tree dose policies from recorded pre-dose histories.\n"
            "- Two folds hold out whole stays; tuning uses training-fitted policies.\n"
            "- Residual dose is a candidate edit feature, not itself proof of causal identification.\n"
            "- GPU factual acceptance threshold remains original NLL + 0.005.\n")
```
