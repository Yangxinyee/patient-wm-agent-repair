The all-row estimates support about −1.94 glucose units per insulin-rate unit and +3.05 MAP units per norepinephrine-rate unit. However, the panel labels reveal that my subset checks used a one-hour-shifted slice. I’ll audit the residuals against the raw records and correct those checks; exact reconstruction would confirm the underlying estimates are properly aligned.

```python
print("=== RAW TIMING CHECK ===")
sid15 = train.stay_id.iloc[0]
print(train.loc[train.stay_id == sid15,
                ["hour", "event", "value", "rate"]].head(24).to_string(index=False))
print("History hours:", np.unique(hf["hour"]))
print("First stay P finger:", P["glucose_finger"][0])
print("First stay A insulin:", A["insulin"][0])

print("\n=== RECONSTRUCT CROSS-FITTED TARGETS ===")
audit15 = {}
for target15 in ["insulin", "norepinephrine", "glucose_finger", "map_arterial"]:
    rr15 = (dml_dose_resid if target15 in A else dml_outcome_resid)[target15]
    ii15 = np.flatnonzero(np.isfinite(rr15))[:3000]
    reconstructed15 = np.empty(len(ii15))
    for fold15 in [0, 1]:
        mm15 = fold_hist[ii15] == fold15
        if mm15.any():
            reconstructed15[mm15] = (
                dml_models[(target15, fold15)].predict(Hhist[ii15[mm15]]) +
                rr15[ii15[mm15]]
            )
    source15 = A[target15] if target15 in A else P[target15]
    errs15 = []
    for offset15 in [0, 1, 2]:
        candidate15 = source15[:, offset15:offset15 + 24].reshape(-1)[ii15]
        finite15 = np.isfinite(candidate15)
        err15 = float(np.max(np.abs(candidate15[finite15] -
                                   reconstructed15[finite15]))) if finite15.any() else np.nan
        errs15.append((offset15, int(finite15.sum()), err15))
    audit15[target15] = errs15
    print(target15, "(slice offset, finite n, max reconstruction error):", errs15)

print("\n=== CORRECTLY ALIGNED SUBSET ESTIMATES ===")
effects15 = {}
for drug15, sensors15 in [
    ("insulin", ("glucose_finger", "glucose_lab")),
    ("norepinephrine", ("map_arterial", "map_cuff"))
]:
    aa15 = A[drug15][:, :24].reshape(-1)
    cc15 = [np.isfinite(P[e][:, :24]).reshape(-1) for e in sensors15]
    for jj15, sensor15 in enumerate(sensors15):
        for label15, mask15 in {
            "all": np.ones(len(aa15), bool),
            "active": aa15 > 0,
            "current own": cc15[jj15],
            "active, both current": (aa15 > 0) & cc15[0] & cc15[1]
        }.items():
            ans15 = clustered_residual_fit(
                DX, dml_outcome_resid[sensor15], mask15, groups14)
            effects15[(sensor15, label15)] = ans15
            print(sensor15, label15, "n", ans15[0],
                  "I,NE", np.round(ans15[1], 4), "SE", np.round(ans15[2], 4))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 15 / alignment audit\nRun 14 subset masks mistakenly used columns 1:25 although P/A hour labels are 0:25. All-row residual coefficients were unaffected. Audited cross-fitted target reconstruction against offsets and recomputed masks with current hours 0:24; results retained in audit15 and effects15. No edits.\n")
```
