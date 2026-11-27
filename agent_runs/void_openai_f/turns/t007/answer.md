Factual evaluation failed on the same missing tuning file, so there is still no forward evidence. The records show almost no same-hour glucose difference between sensors, but cuff MAP averages about 3 mmHg above arterial MAP. I’ll account for that offset and estimate dose responses using within-stay comparisons. Agreement across history and trend controls would strengthen the estimate; disagreement would expose remaining confounding.

```python
import scipy.linalg as sla

torch.set_num_threads(8)
nstay, nhour = Rtrain.stay_id.nunique(), 26
arr = {c: P[c].to_numpy(dtype=np.float64).reshape(nstay, nhour)
       for c in P.columns}

def fuse(x, z):
    count = np.isfinite(x).astype(float) + np.isfinite(z).astype(float)
    return np.divide(np.nan_to_num(x) + np.nan_to_num(z), count,
                     out=np.full_like(x, np.nan), where=count > 0)

Gobs = fuse(arr["glucose_finger"], arr["glucose_lab"])
Mobs = fuse(arr["map_arterial"], arr["map_cuff"] - 3.0)

def panel_regression(y, X, names, hours, mode="fe"):
    n, t = y.shape
    gid = np.repeat(np.arange(n), t)
    hv = np.tile(hours, n).astype(float)
    yy, xx = y.ravel(), X.reshape(n*t, -1)
    good = np.isfinite(yy) & np.isfinite(xx).all(axis=1)
    yy, xx, gid, hv = yy[good], xx[good], gid[good], hv[good]
    if mode in ("fe", "trend"):
        cnt = np.bincount(gid, minlength=n)
        den = np.maximum(cnt, 1)
        mx = np.vstack([np.bincount(gid, weights=xx[:,j], minlength=n)/den
                        for j in range(xx.shape[1])]).T
        my = np.bincount(gid, weights=yy, minlength=n)/den
        xx = xx - mx[gid]
        yy = yy - my[gid]
        if mode == "trend":
            mh = np.bincount(gid, weights=hv, minlength=n)/den
            hc = hv - mh[gid]
            sh = np.bincount(gid, weights=hc*hc, minlength=n)
            denh = np.maximum(sh, 1e-12)
            bx = np.vstack([np.bincount(gid, weights=hc*xx[:,j],
                                        minlength=n)/denh
                            for j in range(xx.shape[1])]).T
            by = np.bincount(gid, weights=hc*yy, minlength=n)/denh
            xx = xx - hc[:,None]*bx[gid]
            yy = yy - hc*by[gid]
    else:
        xx = np.column_stack([np.ones(len(yy)), xx])
        names = ["intercept"] + names
    gram = xx.T @ xx
    inv = np.linalg.pinv(gram, rcond=1e-10)
    coef = inv @ (xx.T @ yy)
    resid = yy - xx @ coef
    scores = np.zeros((n, xx.shape[1]))
    np.add.at(scores, gid, xx*resid[:,None])
    cov = inv @ (scores.T @ scores) @ inv
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    result = pd.DataFrame({"coef": coef, "cluster_se": se}, index=names)
    return result, cov, len(yy), float(np.sqrt(np.mean(resid**2)))

estimates = {}
hours = np.arange(4, 24)
ix = hours + 1
for drug, obs, other in [
    ("insulin", Gobs, "norepinephrine"),
    ("norepinephrine", Mobs, "insulin")
]:
    print("\n===", drug, "NEXT-HOUR RESPONSE ===")
    y = obs[:, ix+1]
    base_names = [f"dose_lag{k}" for k in range(5)] + [
        f"other_lag{k}" for k in range(3)] + ["hour", "hour2"]
    base_cols = [arr[drug][:, ix-k] for k in range(5)] + [
        arr[other][:, ix-k] for k in range(3)] + [
        np.broadcast_to(hours, y.shape), np.broadcast_to(hours**2, y.shape)]
    for label, mode, ar in [
        ("pooled_history", "pooled", True),
        ("within_stay", "fe", False),
        ("within_stay_history", "fe", True),
        ("within_stay_trend_history", "trend", True)
    ]:
        names, cols = list(base_names), list(base_cols)
        if ar:
            names += [f"reading_lag{k}" for k in range(3)]
            cols += [obs[:, ix-k] for k in range(3)]
        fit, cov, n, rmse = panel_regression(
            y, np.stack(cols, axis=-1), names, hours, mode
        )
        estimates[(drug, label)] = (fit, cov, n, rmse)
        sel = [f"dose_lag{k}" for k in range(5)]
        loc = [fit.index.get_loc(k) for k in sel]
        total_se = np.sqrt(max(cov[np.ix_(loc,loc)].sum(), 0))
        print(label, "n =", n, "RMSE =", round(rmse, 3))
        print(fit.loc[sel + ([f"reading_lag{k}" for k in range(3)]
                            if ar else [])].round(4).to_string())
        print("Sum of dose lags:",
              round(fit.loc[sel, "coef"].sum(), 4),
              "+/-", round(1.96*total_se, 4))

rng = np.random.default_rng(4821)
train_probe_ids = rng.choice(np.sort(Rtrain.stay_id.unique()),
                            100, replace=False).tolist()
print("\nTRAIN PROBE IDS:", json.dumps(train_probe_ids))
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Record alignment and confounding\nSame-hour finger/lab difference "
        "is approximately zero (SD 9.93); arterial/cuff difference is -3.01 "
        "(SD 6.72). Raw dose–next-reading associations strongly reflect selection. "
        "Original factual/tuning also failed on missing cuts_tuning.parquet, not counted. "
        "Computed pooled and within-stay distributed-lag regressions, including "
        "history and individual-trend sensitivity checks; results retained in estimates.\n"
        "Python runs: 4. Forward passes: 0.\n"
    )
```
