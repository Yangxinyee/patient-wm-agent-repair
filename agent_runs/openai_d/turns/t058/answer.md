The provisional record-based rollout targets are much larger than the original model’s: roughly −14.2/+7.1 for insulin and +4.7/−2.3 for norepinephrine. I’ll test an insulin correction at the value head, using dose variation not explained by prior condition. It is worth trying only if it predicts held-out innovations and can leave other reading types unchanged.

```python
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, roc_auc_score

print("=== READING METADATA ===")
print(wr_meta.columns.tolist() if hasattr(wr_meta, "columns") else type(wr_meta))
print(wr_meta.head(3).to_string(index=False) if hasattr(wr_meta, "head") else str(wr_meta[:3]))
print("score:", wr_score)
print("instrument keys:", {k: list(v) if isinstance(v, dict) else type(v).__name__
                           for k, v in wr_instruments.items()})

trial_relevant = np.asarray(repair_decoder_cv["insulin"]["relevant"], dtype=bool)
trial_active = np.asarray(repair_decoder_cv["insulin"]["active"], dtype=bool)
trial_y = np.asarray(repair_decoder_cv["insulin"]["target"], dtype=np.float64)
trial_X = np.asarray(rec_features["with_code_no_register_axes"], dtype=np.float64)
trial_groups = np.asarray(rec_head_groups)
trial_rows = np.flatnonzero(trial_relevant)
trial_tr0, trial_va0 = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=719).split(
    trial_rows, groups=trial_groups[trial_rows]))
trial_tr = trial_rows[trial_tr0]
trial_va = trial_rows[trial_va0]
trial_tra = trial_tr[trial_active[trial_tr]]
trial_vaa = trial_va[trial_active[trial_va]]

trial_candidates = []
for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    fit = Ridge(alpha=alpha).fit(trial_X[trial_tra], trial_y[trial_tra])
    pred = fit.predict(trial_X[trial_va])
    result = (
        alpha,
        r2_score(trial_y[trial_vaa], fit.predict(trial_X[trial_vaa])),
        np.sqrt(np.mean(pred[~trial_active[trial_va]] ** 2)),
        np.std(pred)
    )
    trial_candidates.append((result, fit))
print("\n=== ACTIVE-DOSE DECODER ===")
for result, _ in trial_candidates:
    print("alpha, active held-out R2, inactive prediction RMS, prediction SD:", result)
trial_best_result, trial_active_decoder = max(trial_candidates, key=lambda pair: pair[0][1])

print("\n=== ACTIVE-INFUSION CLASSIFICATION ===")
for C in [0.001, 0.01, 0.1, 1]:
    fit = LogisticRegression(C=C, max_iter=500).fit(
        trial_X[trial_tr], trial_active[trial_tr])
    probability = fit.predict_proba(trial_X[trial_va])[:, 1]
    print("C, held-out AUC, accuracy:", C,
          roc_auc_score(trial_active[trial_va], probability),
          np.mean((probability > 0.5) == trial_active[trial_va]))

print("\n=== AVAILABLE HEAD NEURONS ===")
trial_W2 = np.asarray(load_original("attr_heads.value_bin.2.weight"))
trial_phi = np.asarray(wr_phi)
trial_importance = np.std(trial_phi, axis=0) * np.linalg.norm(
    trial_W2 - trial_W2.mean(axis=0, keepdims=True), axis=0)
trial_spare = np.argsort(trial_importance)[:12]
print("lowest-contribution neurons:", list(zip(trial_spare.tolist(),
                                             trial_importance[trial_spare].tolist())))
print("Best active decoder:", trial_best_result)
print("Current edits:", edited())
```
