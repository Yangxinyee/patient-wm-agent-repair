The test stopped because a cached feature matrix contained only 437 rows, rather than all 4,510 readings. No weights changed, and there is no decoder result yet. I’ll rebuild the features from the full cache and repeat the held-out tests. Useful innovation prediction with small inactive-hour errors would support the head correction.

```python
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, roc_auc_score

trial_relevant = np.asarray(repair_decoder_cv["insulin"]["relevant"], dtype=bool)
trial_active = np.asarray(repair_decoder_cv["insulin"]["active"], dtype=bool)
trial_y = np.asarray(repair_decoder_cv["insulin"]["target"], dtype=np.float64)
trial_X = np.asarray(rec_head_X, dtype=np.float64) @ np.asarray(rec_reg_projector)
trial_groups = np.asarray(rec_head_groups)
print("Full shapes:", trial_X.shape, trial_y.shape, trial_active.shape, trial_groups.shape)

trial_rows = np.flatnonzero(trial_relevant)
trial_tr0, trial_va0 = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=719).split(
    trial_rows, groups=trial_groups[trial_rows]))
trial_tr = trial_rows[trial_tr0]
trial_va = trial_rows[trial_va0]
trial_tra = trial_tr[trial_active[trial_tr]]
trial_vaa = trial_va[trial_active[trial_va]]
print("Primary / active / held-out active counts:",
      len(trial_rows), trial_active.sum(), len(trial_vaa))

trial_candidates = []
for alpha in [0.001, 0.01, 0.1, 1, 10, 100, 1000]:
    fit = Ridge(alpha=alpha).fit(trial_X[trial_tra], trial_y[trial_tra])
    pred = fit.predict(trial_X[trial_va])
    inactive = ~trial_active[trial_va]
    result = (
        alpha,
        r2_score(trial_y[trial_vaa], fit.predict(trial_X[trial_vaa])),
        np.sqrt(np.mean(pred[inactive] ** 2)),
        np.std(pred)
    )
    trial_candidates.append((result, fit))
print("\n=== ACTIVE-DOSE DECODER ===")
for result, _ in trial_candidates:
    print("alpha, active R2, inactive RMS, prediction SD:", result)
trial_best_result, trial_active_decoder = max(trial_candidates, key=lambda pair: pair[0][1])

print("\n=== ACTIVE-INFUSION CLASSIFICATION ===")
trial_classifiers = []
for C in [0.001, 0.01, 0.1, 1]:
    fit = LogisticRegression(C=C, max_iter=500).fit(
        trial_X[trial_tr], trial_active[trial_tr])
    probability = fit.predict_proba(trial_X[trial_va])[:, 1]
    result = (C, roc_auc_score(trial_active[trial_va], probability),
              np.mean((probability > 0.5) == trial_active[trial_va]))
    trial_classifiers.append((result, fit))
    print("C, held-out AUC, accuracy:", result)

trial_W2 = np.asarray(load_original("attr_heads.value_bin.2.weight"))
trial_phi = np.asarray(wr_phi)
trial_importance = np.std(trial_phi, axis=0) * np.linalg.norm(
    trial_W2 - trial_W2.mean(axis=0, keepdims=True), axis=0)
trial_spare = np.argsort(trial_importance)[:12]
print("\nLowest-contribution neurons:",
      list(zip(trial_spare.tolist(), trial_importance[trial_spare].tolist())))
print("Best active decoder:", trial_best_result)
print("Current edits:", edited())
```
