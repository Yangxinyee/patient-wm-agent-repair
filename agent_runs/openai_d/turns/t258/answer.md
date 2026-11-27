The norepinephrine pair activates on about half the arterial readings, including early hours. That alone does not show whether it activates before treatment. I’ll partition its factual loss by prior exposure and test an exposure classifier fitted only on training records. A guard is promising only if it separates untreated from treated histories and the untreated portion explains meaningful excess loss.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

r222_meta = {}
r222_rows = []
r222_losses = {}

for split, cc, records in [
    ("train", c210_train, train),
    ("tune100", c210_tune, r72_tune)
]:
    meta = cc["meta"].copy()
    for drug in ["insulin", "norepinephrine"]:
        starts = records.loc[
            (records["event"] == drug) & (records["rate"].fillna(0) > 0),
            ["stay_id", "hour"]
        ].groupby("stay_id")["hour"].min()
        first = meta["stay_id"].map(starts).to_numpy()
        meta[f"{drug}_before"] = np.isfinite(first) & (first < meta["hour"].to_numpy())
        meta[f"{drug}_elapsed"] = meta["hour"].to_numpy() - first
    r222_meta[split] = meta

    losses = {
        name: c210_eval(c206_candidates[name], cc)[0]
        for name in ["original", "insulin_only", "combined_strong"]
    }
    r222_losses[split] = losses
    for drug, event, comparison in [
        ("insulin", "glucose_finger", "insulin_only"),
        ("insulin", "glucose_lab", "insulin_only"),
        ("norepinephrine", "map_arterial", "combined_strong")
    ]:
        baseline = "original" if drug == "insulin" else "insulin_only"
        delta = losses[comparison] - losses[baseline]
        for exposed in [False, True]:
            mask = ((meta["event"].to_numpy() == event)
                    & (meta[f"{drug}_before"].to_numpy() == exposed))
            if mask.any():
                r222_rows.append({
                    "split": split, "drug": drug, "event": event,
                    "previously_exposed": exposed, "n": int(mask.sum()),
                    "mean_delta": float(delta[mask].mean()),
                    "whole_cache_contribution": float(delta[mask].sum() / len(meta))
                })

print("=== LOSS PARTITION ===")
print(pd.DataFrame(r222_rows).round(6).to_string(index=False))

# This is a small statistical classifier, not training the world model.
r222_ne_mask = (
    c210_train["meta"]["event"].to_numpy() == "map_arterial"
)
r222_Xmean = c210_train["X"].mean(0).numpy()[r222_ne_mask]
r222_y = r222_meta["train"]["norepinephrine_before"].to_numpy()[r222_ne_mask].astype(int)
r222_groups = c210_train["meta"]["stay_id"].to_numpy()[r222_ne_mask]
r222_oof = np.zeros(len(r222_y))

for tr, va in GroupKFold(4).split(r222_Xmean, r222_y, r222_groups):
    model222 = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.03, max_iter=500, solver="lbfgs")
    )
    model222.fit(r222_Xmean[tr], r222_y[tr])
    r222_oof[va] = model222.decision_function(r222_Xmean[va])

r222_guard = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.03, max_iter=500, solver="lbfgs")
)
# Include prior-sample variation in the final classifier.
xx222 = c210_train["X"][::4, r222_ne_mask, :].numpy()
r222_guard.fit(
    xx222.reshape(-1, 192),
    np.tile(r222_y, xx222.shape[0])
)
sc222, lr222 = r222_guard.steps[0][1], r222_guard.steps[1][1]
r222_guard_v = (lr222.coef_[0] / sc222.scale_).astype(np.float32)
r222_guard_b = float(lr222.intercept_[0] - r222_guard_v @ sc222.mean_)

print("\n=== TRAINING-ONLY EXPOSURE CLASSIFIER ===")
print("group-held-out AUC:", round(roc_auc_score(r222_y, r222_oof), 5))
for exposed in [0, 1]:
    print("exposed", exposed, "OOF score quantiles:",
          np.round(np.quantile(r222_oof[r222_y == exposed],
                               [0.01, 0.05, 0.25, 0.5, 0.75, 0.95]), 3))
for split, cc in [("train", c210_train), ("tune100", c210_tune)]:
    mask = cc["meta"]["event"].to_numpy() == "map_arterial"
    yy = r222_meta[split]["norepinephrine_before"].to_numpy()[mask].astype(int)
    ss = (cc["X"].mean(0).numpy()[mask] @ r222_guard_v + r222_guard_b)
    print(split, "AUC", round(roc_auc_score(yy, ss), 5),
          "score means", [round(float(ss[yy == j].mean()), 3) for j in [0, 1]])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 222: independent exposure-loss audit\n")
    f.write("No weights changed. Partitioned fresh-cache correction losses by "
            "prior drug exposure and fitted a training-only NE-exposure classifier.\n")
    f.write(pd.DataFrame(r222_rows).to_string(index=False) + "\n")
    f.write("NE group-held-out exposure AUC: "
            + str(roc_auc_score(r222_y, r222_oof)) + "\n")
```
