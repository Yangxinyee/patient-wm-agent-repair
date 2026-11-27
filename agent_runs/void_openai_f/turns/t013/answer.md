Both training and tuning forward requests failed because backend episode files are missing, so factual fit remains untested. The records do contain substantial dose variation. I’ll now estimate one-hour effects using cross-fitted models of strictly pre-dose history, with no weight changes. Stable effects across held-out stays and history lengths would support identification; unstable estimates would not.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time

t0 = time.time()
H = np.sort(P.index.get_level_values("h").unique()).astype(int)
assert np.array_equal(H, np.arange(-1, 25))
nstay = arr["insulin"].shape[0]
causal_hours = np.arange(4, 24)
ci = causal_hours + 1
sid_index = np.repeat(np.arange(nstay), len(ci))
hour_index = np.tile(causal_hours, nstay)

def contemporaneous_mean(a, b, b_offset=0.0):
    a, b = np.asarray(a, float), np.asarray(b, float) + b_offset
    count = np.isfinite(a).astype(int) + np.isfinite(b).astype(int)
    return np.divide(np.nan_to_num(a) + np.nan_to_num(b), count,
                     out=np.full_like(a, np.nan), where=count > 0)

Gsame = contemporaneous_mean(arr["glucose_finger"], arr["glucose_lab"])
Msame = contemporaneous_mean(arr["map_arterial"], arr["map_cuff"], -3.01)
Glast = pd.DataFrame(Gsame).ffill(axis=1).to_numpy()
Mlast = pd.DataFrame(Msame).ffill(axis=1).to_numpy()

feature_arrays, causal_feature_names = [], []
def add_feature(name, a):
    feature_arrays.append(np.asarray(a[:, ci], np.float32).reshape(-1))
    causal_feature_names.append(name)

for code in measurements:
    a = np.asarray(arr[code], float)
    for lag in range(6):
        shifted = np.full_like(a, np.nan)
        if lag == 0:
            shifted[:] = a
        else:
            shifted[:, lag:] = a[:, :-lag]
        add_feature(f"{code}_lag{lag}", shifted)
    last = pd.DataFrame(a).ffill(axis=1).to_numpy()
    age = np.arange(26)[None, :] - np.maximum.accumulate(
        np.where(np.isfinite(a), np.arange(26)[None, :], -30), axis=1)
    add_feature(f"{code}_last", last)
    add_feature(f"{code}_age", age)

for label, a in [("G", Gsame), ("M", Msame)]:
    for width in [4, 8, 26]:
        mean = np.full_like(a, np.nan)
        slope = np.full_like(a, np.nan)
        count = np.zeros_like(a)
        for j in ci:
            jj = np.arange(max(0, j-width+1), j+1)
            v = a[:, jj]
            good = np.isfinite(v)
            n = good.sum(axis=1)
            sx = (good * jj).sum(axis=1)
            sy = np.nan_to_num(v).sum(axis=1)
            sxx = (good * jj**2).sum(axis=1)
            sxy = (np.nan_to_num(v) * jj).sum(axis=1)
            den = n*sxx-sx*sx
            mean[:, j] = np.divide(sy, n, out=np.full(nstay, np.nan), where=n>0)
            slope[:, j] = np.divide(n*sxy-sx*sy, den,
                                    out=np.zeros(nstay), where=den>0)
            count[:, j] = n
        add_feature(f"{label}_mean{width}", mean)
        add_feature(f"{label}_slope{width}", slope)
        add_feature(f"{label}_count{width}", count)

for drug in drugs:
    a = np.asarray(arr[drug], float)
    for lag in range(1, 7):
        shifted = np.zeros_like(a)
        shifted[:, lag:] = a[:, :-lag]
        add_feature(f"{drug}_lag{lag}", shifted)

Xhistory = np.column_stack(feature_arrays + [hour_index.astype(np.float32)])
causal_feature_names.append("hour")
fold_by_stay = np.random.default_rng(913).permutation(nstay) % 2
fold_rows = fold_by_stay[sid_index]
crossfit = {}

def cluster_slope(d, y, groups):
    den = np.dot(d, d)
    beta = np.dot(d, y) / den
    scores = np.bincount(groups, weights=d*(y-beta*d), minlength=nstay)
    se = np.sqrt(np.dot(scores, scores)) / den
    return float(beta), float(se)

with threadpool_limits(limits=8):
    for drug, outcome, baseline in [
        ("insulin", "glucose_finger", Glast),
        ("norepinephrine", "map_arterial", Mlast)
    ]:
        other = "norepinephrine" if drug == "insulin" else "insulin"
        X = np.column_stack([Xhistory, arr[other][:, ci].reshape(-1)]).astype(np.float32)
        d = arr[drug][:, ci].reshape(-1).astype(float)
        y = (arr[outcome][:, ci+1] - baseline[:, ci]).reshape(-1)
        valid = np.isfinite(y)
        pdose = np.full(len(d), np.nan)
        pout = np.full(len(d), np.nan)
        fitted = []
        for fold in [0, 1]:
            train = valid & (fold_rows != fold)
            test = valid & (fold_rows == fold)
            kwargs = dict(max_iter=110, max_leaf_nodes=23, min_samples_leaf=150,
                          learning_rate=.08, l2_regularization=20,
                          early_stopping=False, random_state=100+fold)
            md = HistGradientBoostingRegressor(**kwargs)
            my = HistGradientBoostingRegressor(**kwargs)
            md.fit(X[train], d[train])
            my.fit(X[train], y[train])
            pdose[test] = md.predict(X[test])
            pout[test] = my.predict(X[test])
            fitted.append((md, my))
            print(drug, "completed fold", fold, "elapsed_s", round(time.time()-t0, 1))
        dr, yr = d-pdose, y-pout
        crossfit[drug] = dict(d=d, y=y, dr=dr, yr=yr, pdose=pdose,
                             pout=pout, valid=valid, fitted=fitted)
        print("\n", drug, "one-hour effect per rate unit:")
        for label, sub in [
            ("all eligible hours", valid),
            ("hours 12-23", valid & (hour_index >= 12)),
            ("positive recorded dose (sensitivity)", valid & (d > 0))
        ]:
            beta, se = cluster_slope(dr[sub], yr[sub], sid_index[sub])
            print(label, "beta", round(beta, 5), "cluster_SE", round(se, 5),
                  "N", int(sub.sum()), "dose_residual_SD", round(np.std(dr[sub]), 5))
        print("Outcome residual SD:", round(np.std(yr[valid]), 4))
        print("Dose prediction R2:",
              round(1-np.mean(dr[valid]**2)/np.var(d[valid]), 5))

print("\nFeatures:", len(causal_feature_names)+1, "total elapsed_s", round(time.time()-t0, 1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Cross-fitted causal analysis\nBuilt features using only measurements and drug histories available before each action, including multi-sensor levels, ages, and trailing trends. Two folds split by stay; nuisance models fitted separately from world-model weights. Results retained in crossfit. Forward backend missing cuts files for both splits; no factual/rollout validation available yet. Python runs: 9; successful forward passes: 0.\n")
```