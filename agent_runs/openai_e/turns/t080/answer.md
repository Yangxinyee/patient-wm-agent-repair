The recovered +3.25 norepinephrine estimate is still provisional. I’ll cross-check it using separate arterial and cuff outcomes and a pre-treatment placebo, without changing weights. Agreement on future MAP and little association with earlier MAP would support a targeted repair; otherwise, I’ll investigate the remaining confounding.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from pathlib import Path

rng69 = np.random.default_rng(6901)
eligible69 = panel.index[(panel["t"] >= 0) & (panel["t"] <= 23)]
sample_idx69 = rng69.choice(eligible69.to_numpy(),
                           min(160000, len(eligible69)), replace=False)
s69 = panel.loc[sample_idx69].copy().reset_index(drop=True)
groups69 = panel.groupby("stay_id", sort=False)

targets69 = {
    name + "_next": groups69[name].shift(-1).loc[sample_idx69].to_numpy()
    for name in ("map_arterial", "map_cuff", "map_any")
}
targets69["map_five_hours_before"] = (
    groups69["map_any"].shift(5).loc[sample_idx69].to_numpy()
)
features69 = nuisance_features(s69)
X69 = np.asarray(features69[0] if isinstance(features69, tuple)
                 else features69, dtype=np.float32)
X69 = np.column_stack((X69, s69["insulin"].to_numpy(dtype=np.float32)))
d69 = s69["norepinephrine"].to_numpy(dtype=np.float64)
fold69 = (s69["stay_id"].to_numpy(dtype=np.int64) % 2).astype(int)
settings69 = dict(max_iter=120, learning_rate=0.08, max_leaf_nodes=31,
                  max_bins=128, min_samples_leaf=100,
                  l2_regularization=10, early_stopping=False,
                  random_state=6901)

pred_d69 = np.full(len(s69), np.nan)
pred_y69 = {name: np.full(len(s69), np.nan) for name in targets69}
models69 = {"dose": []}
with threadpool_limits(limits=8):
    for f69 in (0, 1):
        train69, test69 = fold69 != f69, fold69 == f69
        model69 = HistGradientBoostingRegressor(**settings69)
        model69.fit(X69[train69], d69[train69])
        pred_d69[test69] = model69.predict(X69[test69])
        models69["dose"].append(model69)
        for name69, outcome69 in targets69.items():
            observed69 = np.isfinite(outcome69)
            model69 = HistGradientBoostingRegressor(**settings69)
            model69.fit(X69[train69 & observed69],
                        outcome69[train69 & observed69])
            pred_y69[name69][test69 & observed69] = model69.predict(
                X69[test69 & observed69])
            models69.setdefault(name69, []).append(model69)

resid_d69 = d69 - pred_d69

def innovation_estimate69(outcome, prediction, mask=None):
    ok = np.isfinite(outcome) & np.isfinite(prediction)
    if mask is not None:
        ok &= mask
    instrument = resid_d69[ok]
    residual = outcome[ok] - prediction[ok]
    denominator = np.dot(instrument, instrument)
    beta = np.dot(instrument, residual) / denominator
    scores = instrument * (residual - beta * instrument)
    _, inverse = np.unique(s69.loc[ok, "stay_id"], return_inverse=True)
    cluster_scores = np.bincount(inverse, weights=scores)
    n_clusters = len(cluster_scores)
    se = np.sqrt(n_clusters / (n_clusters - 1) *
                 np.dot(cluster_scores, cluster_scores)) / denominator
    return dict(n=int(ok.sum()), stays=n_clusters, beta=float(beta),
                cluster_se=float(se), dose_residual_sd=float(instrument.std()))

estimates69 = {
    name: innovation_estimate69(outcome, pred_y69[name])
    for name, outcome in targets69.items()
}
print("Dose prediction R2:",
      1 - np.mean(resid_d69 ** 2) / np.var(d69))
print(pd.DataFrame(estimates69).T.to_string())

def record_schedule_target69(beta, phi, drug, edit_name):
    responses = {}
    for stay, frame in tpanel.groupby("stay_id", sort=False):
        rates = frame.set_index("t")[drug].to_dict()
        response = 0.0
        for hour in range(13, 25):
            response = phi * response + beta * float(rates.get(hour - 1, 0.0))
            responses[(int(stay), hour)] = response
    rows = rollout68e.loc[
        (rollout68e["edit"] == edit_name) & rollout68e["reached"].astype(bool)]
    values = [responses[(int(row.stay_id), int(row.hour))]
              for row in rows.itertuples()]
    return dict(mean=float(np.mean(values)), n=len(values))

print("Insulin x2 record-based target:",
      record_schedule_target69(float(dyn_fit.x[0]), float(dyn_fit.x[1]),
                               "insulin", "insulin_x2"))
print("NE x2 persistence sensitivity, provisional beta=3.25:")
for phi69 in (0.0, 0.4, 0.6, 0.75, 0.85, 0.95):
    print(phi69, record_schedule_target69(3.25, phi69,
                                         "norepinephrine", "ne_x2"))

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 69: independent NE identification audit\n"
            "Cross-fitted nuisance regressions on 160000 training intervals, split by stay. "
            "Compared next arterial, cuff and pooled MAP and a five-hour pre-treatment placebo. "
            "Estimates: " + repr(estimates69) +
            ". Computed schedule-target sensitivity on forward-10 reached hours. "
            "No weights changed. Python runs: 69; forwards: 10.\n")
```