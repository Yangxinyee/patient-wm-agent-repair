The pooled refit has lower held-out residuals on the independent training states, but comparison with the old decoder is partly in-sample. I’ll check the old fitting procedure under the same folds, then provisionally install the matched-sensitivity NE refit. A factual test on the cohort that previously exceeded the budget will decide whether it is useful.

```python
import torch, hashlib

old_procedure_cv204 = np.zeros(len(X147))
for fold204 in range(5):
    ft204 = fact_folds172 != fold204
    pt204 = probe_folds172 != fold204
    xf204, dp204, yp204 = X147[ft204], D147[pt204], Y147[pt204]
    b204 = fit_zero_fact171(
        xf204.T @ xf204 / len(xf204),
        dp204.T @ dp204 / len(dp204),
        dp204.T @ yp204 / len(dp204),
        0.03, 1e-6
    )
    nm204 = ne_double202 & pt204
    b204[:,1] *= (
        np.mean((D147 @ Bcurrent202)[nm204,1]) /
        np.mean((D147 @ b204)[nm204,1])
    )
    old_procedure_cv204[~ft204] = X147[~ft204] @ b204[:,1]

print("MATCHED OLD-COHORT CV RMS: OLD PROCEDURE / POOLED REFIT",
      rms202(old_procedure_cv204),
      rms202(cv_predictions203[(0.003,1e-7)]["old"]))

candidate_key204 = (0.003, 1e-7)
candidate_W204 = fits202[candidate_key204][1].copy()
candidate_W204[:,0] = validated_W176[:,0]
original_B204 = np.asarray(load_original("core.b_in.weight"), dtype=np.float64)
print("EXACT EDIT-DIRECTION RECONSTRUCTION ERROR",
      np.max(np.abs(validated_B176 -
                    (original_B204 + r199_L @ validated_W176.T))))

candidate_B204 = (
    np.asarray(validated_B176, dtype=np.float64)
    + np.outer(r199_L[:,1],
               candidate_W204[:,1] - validated_W176[:,1])
).astype(np.float32)
save_edit("core.b_in.weight", torch.from_numpy(candidate_B204))
print("PROVISIONAL EDIT HASH",
      hashlib.sha256(candidate_B204.tobytes()).hexdigest()[:12])
print("ONLY EDITED TENSOR", edited())
print("INSULIN DECODER UNCHANGED",
      np.array_equal(candidate_W204[:,0], validated_W176[:,0]))

f23_original204 = pd.read_csv("/work/fwd_23/factual_original.csv")
print("FACTUAL COLUMNS", f23_original204.columns.tolist())
stay_col204 = next(c for c in ["stay_id","stay"] if c in f23_original204.columns)
test_stays204 = sorted(f23_original204[stay_col204].unique().tolist())
print("DIFFICULT VALIDATION COHORT", test_stays204)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run204: provisional independent-state NE refit\n"
            "Installed only core.b_in.weight: insulin decoder unchanged; "
            "NE decoder from pooled old/new factual covariance, key "
            "(0.003,1e-7), matched to validated_B176 doubled-dose probe mean. "
            "Validated_B176 is the exact fallback. Must test factual NLL "
            "on the previously difficult third tuning cohort before acceptance.\n")
```
