The state file is available, while the threshold audit still leaves norepinephrine uncertain. I’ll use richer pre-dose histories to check whether the earlier effect estimates were driven by omitted trends or stale readings, and inspect what the states retain about doses. Stability across sensors and increasingly detailed histories would strengthen an estimate; large changes would weaken it.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

S = np.load("/work/fwd_3/states_original.npz")
print("=== STATES ===")
print({k: (S[k].shape, str(S[k].dtype)) for k in S.files})
state_meta = pd.DataFrame({k: S[k] for k in S.files if k != "x"})
print(state_meta.head(12).to_string(index=False))

print("\n=== PAIRED SENSOR DIFFERENCES ===")
for a, b, drug in [("glucose_finger", "glucose_lab", "insulin"),
                   ("map_arterial", "map_cuff", "norepinephrine")]:
    ok = panel[a].notna() & panel[b].notna()
    pair = panel.loc[ok, [a, b, drug]].copy()
    pair["difference"] = pair[a] - pair[b]
    pair["active"] = pair[drug] > 0
    print(a, "-", b)
    print(pair.groupby("active")["difference"].agg(["count", "mean", "std"]).round(4).to_string())

print("\n=== REMAINING CORE/HEAD NAMES ===")
for n in NAMES:
    if any(s in n for s in ["z_prior", "z_post", "core.gap", "gap.", "last_status",
                             "embed.bins", "embed.categorical", "filter", "z_in"]):
        print(n, SHAPES[n])

F = pd.DataFrame(index=panel.index)
F["hour"] = panel.index.get_level_values("t").to_numpy()
base_codes = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
for c in base_codes:
    raw = panel[c]
    ff = raw.groupby(level=0, sort=False).ffill()
    F[c + "_raw"] = raw
    F[c + "_ff"] = ff
    F[c + "_seen"] = raw.notna().astype(float)
    for lag in [1, 2, 3]:
        F[c + f"_lag{lag}"] = raw.groupby(level=0, sort=False).shift(lag)
        F[c + f"_fflag{lag}"] = ff.groupby(level=0, sort=False).shift(lag)
    initial = raw.loc[(slice(None), [-1, 0])].groupby(level=0).mean()
    F[c + "_initial"] = panel.index.get_level_values(0).map(initial).to_numpy()
for c in ["insulin", "norepinephrine"]:
    F[c + "_current"] = panel[c]
    for lag in [1, 2, 3]:
        F[c + f"_lag{lag}"] = panel[c].groupby(level=0, sort=False).shift(lag)
F = F.astype(np.float32)

def rich_dml(drug, outcome, label, extra=None, max_rows=250000):
    start = time.time()
    yy = panel[outcome].groupby(level=0, sort=False).shift(-1)
    eligible = yy.notna().to_numpy() & F.hour.between(1, 23).to_numpy()
    if extra is not None:
        eligible = eligible & np.asarray(extra, bool)
    rows = np.flatnonzero(eligible)
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(835).choice(rows, max_rows, replace=False))
    X = F.drop(columns=[drug + "_current"]).iloc[rows].to_numpy()
    y = yy.iloc[rows].to_numpy(float)
    d = panel[drug].iloc[rows].to_numpy(float)
    ids = panel.index.get_level_values(0).to_numpy()[rows]
    folds = ((ids * 17 + 3) % 2).astype(int)
    ry, rd = np.empty(len(rows)), np.empty(len(rows))
    for fold in [0, 1]:
        tr, te = folds != fold, folds == fold
        settings = dict(max_iter=200, learning_rate=.065, max_leaf_nodes=31,
                        min_samples_leaf=80, l2_regularization=2,
                        early_stopping=False, random_state=13)
        with threadpool_limits(limits=8):
            md = HistGradientBoostingRegressor(**settings).fit(X[tr], d[tr])
            my = HistGradientBoostingRegressor(**settings).fit(X[tr], y[tr])
            rd[te] = d[te] - md.predict(X[te])
            ry[te] = y[te] - my.predict(X[te])
    den = rd @ rd
    theta = (rd @ ry) / den
    _, groups = np.unique(ids, return_inverse=True)
    score = np.bincount(groups, weights=rd * (ry - theta * rd)) / den
    result = dict(label=label, n=len(rows), effect=float(theta),
                  cluster_se=float(np.sqrt(score @ score)),
                  dose_residual_sd=float(rd.std()),
                  outcome_residual_sd=float(ry.std()),
                  seconds=round(time.time() - start, 1))
    return result, dict(rows=rows, ry=ry, rd=rd, ids=ids)

rich_results = []
rich_fits = {}
both_map = panel.map_arterial.notna() & panel.map_cuff.notna()
for drug, outcome, label, extra in [
    ("norepinephrine", "map_cuff", "NE_cuff_both_current", both_map),
    ("norepinephrine", "map_arterial", "NE_arterial_both_current", both_map),
    ("norepinephrine", "map_cuff", "NE_cuff_all_histories", None),
    ("insulin", "glucose_lab", "insulin_lab_all_histories", None)
]:
    result, fit = rich_dml(drug, outcome, label, extra)
    rich_results.append(result)
    rich_fits[label] = fit
print("\n=== RICH HISTORY DML ===")
print(pd.DataFrame(rich_results).round(5).to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 11: rich pre-dose history, stay-crossfitted estimates\n")
    f.write(pd.DataFrame(rich_results).round(5).to_string(index=False) + "\n")
    f.write("States from 100 randomly sampled training stays loaded as S; history features F.\n")
```
