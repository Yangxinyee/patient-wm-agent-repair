The GELU/split prior’s mean error is consistent with four-draw sampling noise, though that does not uniquely verify the packing. I’ll now fit dose-innovation decoders and infusion gates on sampled head inputs, holding out whole stays. Positive held-out decoding and reliable inactive-hour suppression would support a small correction; failure would rule it out.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from threadpoolctl import threadpool_limits

# Project out every varying reading-register direction.
A99 = torch.cat([
    W96["last_numeric.weight"][:, [0, 3, 6]],
    (W96["embed.bins.value.weight"][1:] -
     W96["embed.bins.value.weight"][0]).T
], dim=1).numpy().astype(np.float64)
U99, S99, _ = np.linalg.svd(A99, full_matrices=False)
rank99 = int((S99 > S99[0] * 1e-7).sum())
P99 = np.eye(192) - U99[:, :rank99] @ U99[:, :rank99].T
print("Register projection rank:", rank99)

M99 = cache98["with_code"].numpy()
groups99 = meta96["stay_id"].to_numpy()
events99 = meta96["event"].to_numpy()
split99 = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=9901)
tr99, va99 = next(split99.split(meta96, groups=groups99))
trainmask99 = np.zeros(len(meta96), bool)
trainmask99[tr99] = True
valmask99 = ~trainmask99
fitdraws99 = np.array([0, 3, 7, 11, 16, 19, 23, 27])

assert hasattr(train, "columns") and {"stay_id", "hour", "event", "rate"} <= set(train.columns)
actions99 = train.loc[
    train["stay_id"].isin(np.unique(groups99)) &
    train["event"].isin(["insulin", "norepinephrine"]),
    ["stay_id", "hour", "event", "rate"]
].copy()
actions99["start_hour"] = np.floor(actions99["hour"] + 1e-6).astype(int)
lookup99 = {
    (int(s), int(h), e): float(r)
    for s, h, e, r in actions99[["stay_id", "start_hour", "event", "rate"]].itertuples(index=False, name=None)
}
rates99 = {
    drug: np.array([
        lookup99.get((int(s), int(round(h)) - 1, drug), 0.)
        for s, h in meta96[["stay_id", "hour"]].itertuples(index=False, name=None)
    ])
    for drug in ["insulin", "norepinephrine"]
}

features99 = {}
for kind99, proj99 in [("raw", np.eye(192)), ("register_free", P99)]:
    X99 = (M99 @ proj99).astype(np.float32)
    mean99 = X99[fitdraws99][:, trainmask99].reshape(-1, 192).mean(0)
    sd99 = X99[fitdraws99][:, trainmask99].reshape(-1, 192).std(0).clip(.05)
    features99[kind99] = {
        "X": (X99 - mean99) / sd99, "mean": mean99,
        "sd": sd99, "projection": proj99
    }

fits99, gates99, results99 = {}, {}, []
with threadpool_limits(limits=8):
    for drug99, family99 in [
        ("insulin", ["glucose_finger", "glucose_lab"]),
        ("norepinephrine", ["map_arterial", "map_cuff"])
    ]:
        aligned99 = repair_aligned[drug99]
        assert np.array_equal(aligned99["stay_id"].to_numpy(), groups99)
        assert np.array_equal(aligned99["event"].to_numpy(), events99)
        target99 = aligned99["rd"].to_numpy(dtype=float)
        fam99 = np.isin(events99, family99)
        active99 = rates99[drug99] > 0
        eligible99 = fam99 & active99 & np.isfinite(target99)
        print("\n", drug99, "eligible rows train/validation:",
              int((eligible99 & trainmask99).sum()),
              int((eligible99 & valmask99).sum()),
              "active family rows:", int((fam99 & active99).sum()))

        candidates99 = []
        for kind99, feat99 in features99.items():
            XX99 = feat99["X"]
            ti99 = np.flatnonzero(eligible99 & trainmask99)
            vi99 = np.flatnonzero(eligible99 & valmask99)
            xtrain99 = XX99[fitdraws99][:, ti99].reshape(-1, 192)
            ytrain99 = np.tile(target99[ti99], len(fitdraws99))
            for alpha99 in [.1, 1., 10., 100., 1000., 10000.]:
                fit99 = Ridge(alpha=alpha99).fit(xtrain99, ytrain99)
                pred99 = fit99.predict(XX99[:, vi99].reshape(-1, 192)).reshape(len(M99), -1)
                single_r299 = r2_score(np.tile(target99[vi99], len(M99)), pred99.ravel())
                mean_r299 = r2_score(target99[vi99], pred99.mean(0))
                row99 = (single_r299, mean_r299, kind99, alpha99, fit99)
                candidates99.append(row99)
            gate_target99 = (fam99 & active99).astype(int)
            gate99 = LogisticRegression(C=1., max_iter=600).fit(
                XX99[fitdraws99][:, trainmask99].reshape(-1, 192),
                np.tile(gate_target99[trainmask99], len(fitdraws99)))
            gate_scores99 = gate99.decision_function(
                XX99[:, valmask99].reshape(-1, 192))
            gate_truth99 = np.tile(gate_target99[valmask99], len(M99))
            print(kind99, "gate balanced accuracy:",
                  balanced_accuracy_score(gate_truth99, gate_scores99 > 0),
                  "positive score quantiles:",
                  np.quantile(gate_scores99[gate_truth99 == 1], [.01, .1, .5]),
                  "negative score quantiles:",
                  np.quantile(gate_scores99[gate_truth99 == 0], [.5, .9, .99]))
            gates99[(drug99, kind99)] = gate99

        candidates99.sort(key=lambda x: x[0], reverse=True)
        for row99 in candidates99[:4]:
            print("decoder: single-draw R2, mean-draw R2, feature, alpha:", row99[:4])
        fits99[drug99] = candidates99
        results99.append((drug99, candidates99[0][:4]))

print("CURRENT EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 99: latent-aware, cross-stay decoder tests\n")
    f.write("Physical active-infusion labels recovered directly from records. "
            "Fitting innovation regressions and family/infusion gates on individual "
            "prior draws; holding out 25% of stays. No weight edits.\n")
    f.write(repr(results99) + "\n")
```
