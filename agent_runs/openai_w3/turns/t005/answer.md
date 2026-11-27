The original model predicts almost no insulin response and small, non-monotonic norepinephrine responses. That could be correct or could reflect missed drug action. I’ll estimate next-hour effects after adjusting for recorded history, using stay-separated fitting. Agreement across adjustment methods and measurement channels would strengthen the evidence; disagreement would weaken it.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time
threadpool_limits(limits=8)
torch.set_num_threads(8)
started = time.time()

print("=== PAIRED SENSOR DIFFERENCES ===")
for e1, e2 in [("glucose_finger", "glucose_lab"), ("map_arterial", "map_cuff")]:
    ok = np.isfinite(A[e1]) & np.isfinite(A[e2])
    diff = (A[e1] - A[e2])[ok]
    print(e1, "-", e2, "n:", len(diff), "mean:", round(diff.mean(), 4),
          "sd:", round(diff.std(), 4))

# Every example is a stay-hour t=0..23, before the dose at t+0.0833.
n_stays, nt = len(train_ids), 24
sid_rows = np.repeat(np.arange(n_stays), nt)
hour_rows = np.tile(np.arange(nt), n_stays)
features, feature_names = [], []
def add_feature(name, values):
    features.append(np.asarray(values, dtype=np.float32).reshape(-1))
    feature_names.append(name)

add_feature("hour", np.broadcast_to(np.arange(nt), (n_stays, nt)))
for event in events6[:4]:
    arr = A[event]
    for lag in range(5):
        v = np.full((n_stays, nt), np.nan)
        for t in range(nt):
            j = t + 1 - lag
            if j >= 0:
                v[:, t] = arr[:, j]
        add_feature(event + "_lag" + str(lag), v)
    last = np.full(n_stays, np.nan)
    age = np.full(n_stays, 30.0)
    total = np.zeros(n_stays)
    count = np.zeros(n_stays)
    lasts, ages, means, counts = [], [], [], []
    for j in range(25):
        seen = np.isfinite(arr[:, j])
        age += 1
        last[seen] = arr[seen, j]
        age[seen] = 0
        total[seen] += arr[seen, j]
        count[seen] += 1
        if j >= 1:
            lasts.append(last.copy())
            ages.append(age.copy())
            means.append(np.divide(total, count, out=np.full(n_stays, np.nan),
                                   where=count > 0))
            counts.append(count.copy())
    add_feature(event + "_last", np.stack(lasts, axis=1))
    add_feature(event + "_age", np.stack(ages, axis=1))
    add_feature(event + "_histmean", np.stack(means, axis=1))
    add_feature(event + "_histcount", np.stack(counts, axis=1))
    add_feature(event + "_preadmission", np.broadcast_to(arr[:, :1], (n_stays, nt)))

for event in events6[-2:]:
    arr = A[event]
    for lag in range(1, 5):
        v = np.zeros((n_stays, nt))
        for t in range(nt):
            j = t + 1 - lag
            if j >= 0:
                v[:, t] = arr[:, j]
        add_feature(event + "_lag" + str(lag), v)
    add_feature(event + "_pastsum", np.cumsum(arr, axis=1)[:, :24])

Xhist = np.column_stack(features)
print("\nHistory feature matrix:", Xhist.shape)
rng = np.random.default_rng(412)
fold_by_stay = rng.integers(0, 2, size=n_stays)
fold_rows = fold_by_stay[sid_rows]

def cluster_estimate(rd, ry, ids):
    denom = np.dot(rd, rd)
    beta = np.dot(rd, ry) / denom
    scores = np.bincount(ids, weights=rd * (ry - beta * rd), minlength=n_stays)
    se = np.sqrt(np.dot(scores, scores)) / denom
    return float(beta), float(se)

dml_results = {}
for drug, outcome in [("insulin", "glucose_finger"),
                      ("norepinephrine", "map_arterial")]:
    dose = A[drug][:, 1:25].reshape(-1)
    next_y = A[outcome][:, 2:26].reshape(-1)
    ok = np.isfinite(next_y)
    rows = np.flatnonzero(ok)
    xx, dd, yy = Xhist[rows], dose[rows], next_y[rows]
    pred_d, pred_y = np.zeros(len(rows)), np.zeros(len(rows))
    for fold in [0, 1]:
        fit = fold_rows[rows] != fold
        test = ~fit
        params = dict(max_iter=110, max_leaf_nodes=23, learning_rate=.09,
                      max_bins=128, min_samples_leaf=160, l2_regularization=5,
                      early_stopping=False, random_state=97 + fold)
        md = HistGradientBoostingRegressor(**params).fit(xx[fit], dd[fit])
        my = HistGradientBoostingRegressor(**params).fit(xx[fit], yy[fit])
        pred_d[test] = md.predict(xx[test])
        pred_y[test] = my.predict(xx[test])
    rd, ry = dd - pred_d, yy - pred_y
    beta, se = cluster_estimate(rd, ry, sid_rows[rows])
    positive = dd > 0
    bp, sp = cluster_estimate(rd[positive], ry[positive], sid_rows[rows][positive])
    dml_results[(drug, outcome)] = {
        "rows": rows, "pred_d": pred_d, "pred_y": pred_y,
        "dose_residual": rd, "outcome_residual": ry,
        "beta": beta, "se": se
    }
    print("\n", drug, "->", outcome, "n:", len(rows))
    print("History-adjusted effect/unit:", round(beta, 5), "cluster SE:", round(se, 5))
    print("Positive-dose subset effect/unit:", round(bp, 5), "SE:", round(sp, 5))
    print("Dose residual SD:", round(rd.std(), 5),
          "dose R2:", round(1 - np.mean(rd**2) / np.var(dd), 5),
          "next-outcome residual SD:", round(ry.std(), 5))
    for lo, hi in [(0, 6), (6, 12), (12, 18), (18, 24)]:
        sel = (hour_rows[rows] >= lo) & (hour_rows[rows] < hi)
        b, s = cluster_estimate(rd[sel], ry[sel], sid_rows[rows][sel])
        print("hours", lo, "to", hi-1, "effect:", round(b, 4), "SE:", round(s, 4))

print("\nElapsed seconds:", round(time.time() - started, 1))
with Path("/work/notes.md").open("a") as f:
    f.write("\nPython runs: 4; forward requests: 1. Original rollout on tuning IDs "
            "2020000000..99: insulin x2=-0.422, half=+0.213 glucose units; "
            "NE x2=-0.742, half=-0.238 MAP units. "
            "Created Xhist (recorded pre-dose history), stay-separated twofold "
            "histogram-regression residual estimates of immediate dose effects. "
            "Results: " + str({str(k): {"beta": v["beta"], "se": v["se"]}
                               for k, v in dml_results.items()}) + "\n")
```
