Readings occur on the hour, and infusions start five minutes later. That gives a clear temporal order: readings at hour \(t\), then treatment, then readings at \(t+1\). I’ll compare raw dose associations with history-adjusted and within-stay estimates. Agreement across these checks would support a drug response; disagreement would warn of confounding.

```python
RUNS += 1
events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
          "insulin", "norepinephrine"]
stay_ids = np.sort(train.stay_id.unique())
hours = np.arange(-1, 25)
idx = pd.MultiIndex.from_product([stay_ids, hours], names=["stay_id", "t"])
sub = train[train.event.isin(events)].copy()
sub["t"] = np.floor(sub.hour + 1e-5).astype(int)
vals = sub.pivot(index=["stay_id", "t"], columns="event", values="value").reindex(idx)
rates = sub.pivot(index=["stay_id", "t"], columns="event", values="rate").reindex(idx)
A = {e: vals[e].to_numpy().reshape(len(stay_ids), 26).astype(np.float64)
     for e in events[:4]}
A.update({e: rates[e].fillna(0).to_numpy().reshape(len(stay_ids), 26).astype(np.float64)
          for e in events[4:]})
del sub, vals, rates

def forward_fill(a, fallback):
    good = np.isfinite(a)
    pos = np.maximum.accumulate(np.where(good, np.arange(a.shape[1]), -1), axis=1)
    out = np.take_along_axis(a, np.maximum(pos, 0), axis=1)
    return np.where(pos >= 0, out, fallback)

FF = {e: forward_fill(A[e], np.nanmedian(A[e])) for e in events[:4]}
print("Measurement coverage by hour (-1,0,6,12,24):")
for e in events[:4]:
    print(e, np.round(np.isfinite(A[e][:, [0,1,7,13,25]]).mean(axis=0), 3))
print("\nSimultaneous sensor differences:")
for e1, e2, drug in [("glucose_finger", "glucose_lab", "insulin"),
                      ("map_cuff", "map_arterial", "norepinephrine")]:
    diff = A[e1] - A[e2]
    sm = np.nanmean(diff, axis=1)
    valid = np.isfinite(diff)
    print(e1, "-", e2,
          "mean", round(np.nanmean(diff), 3),
          "SD", round(np.nanstd(diff), 3),
          "SD stay mean", round(np.nanstd(sm), 3),
          "within-stay SD", round(np.sqrt(np.nanmean((diff-sm[:,None])**2)), 3),
          "corr with same-hour dose", round(np.corrcoef(diff[valid], A[drug][valid])[0,1], 3))

# Treatments at t=0..23; outcomes at t+1=1..24.
I = A["insulin"][:, 1:25]
N = A["norepinephrine"][:, 1:25]
GID = np.repeat(np.arange(len(stay_ids)), 24)
T = np.tile(np.arange(24), len(stay_ids))
basic_names = ["intercept", "insulin", "norepinephrine"]
basic = [np.ones(I.size), I.ravel(), N.ravel()]
history_names, history = [], []
for e in events[:4]:
    for lag in [0, 1, 2]:
        z = FF[e][:, max(0,1-lag):25-lag]
        if z.shape[1] < 24:
            z = np.column_stack([FF[e][:,0], z])[:, -24:]
        history.append(z.ravel())
        history_names.append(f"{e}_lag{lag}")
    history.append(np.isfinite(A[e][:,1:25]).ravel().astype(float))
    history_names.append(f"{e}_observed")
for e in events[4:]:
    history.append(A[e][:,0:24].ravel())
    history_names.append(f"{e}_lag1")
for h in range(1,24):
    history.append((T == h).astype(float))
    history_names.append(f"hour{h}")

Xhistory = np.column_stack(basic + history)

def ols_cluster(X, y, groups, within=False):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    groups = np.asarray(groups, dtype=int)
    if within:
        cnt = np.bincount(groups, minlength=len(stay_ids))
        den = np.maximum(cnt, 1)
        mx = np.column_stack([np.bincount(groups, weights=X[:,j],
                                         minlength=len(stay_ids))/den
                              for j in range(X.shape[1])])
        my = np.bincount(groups, weights=y, minlength=len(stay_ids))/den
        X = X-mx[groups]
        y = y-my[groups]
    scale = np.sqrt(np.mean(X*X, axis=0))
    keep = scale > 1e-9
    Xs = X[:,keep]/scale[keep]
    gram = Xs.T@Xs
    inv = np.linalg.pinv(gram, rcond=1e-10)
    beta_s = inv@(Xs.T@y)
    residual = y-Xs@beta_s
    scores = np.column_stack([
        np.bincount(groups, weights=Xs[:,j]*residual, minlength=len(stay_ids))
        for j in range(Xs.shape[1])
    ])
    cov_s = inv@(scores.T@scores)@inv
    beta = np.zeros(X.shape[1]); se = np.zeros(X.shape[1])
    beta[keep] = beta_s/scale[keep]
    se[keep] = np.sqrt(np.maximum(np.diag(cov_s),0))/scale[keep]
    return beta, se, float(np.sqrt(np.mean(residual**2))), len(y)

print("\nInitial dose coefficients: coefficient (cluster SE), residual SD, n")
initial_fits = {}
for e in events[:4]:
    yn = A[e][:,2:26].ravel()
    yc = A[e][:,1:25].ravel()
    for label, XX, yy, within in [
        ("raw next level", np.column_stack(basic), yn, False),
        ("raw change", np.column_stack(basic), yn-yc, False),
        ("history-adjusted", Xhistory, yn, False),
        ("history + stay FE", Xhistory, yn, True)
    ]:
        ok = np.isfinite(yy)
        b,s,sd,n = ols_cluster(XX[ok], yy[ok], GID[ok], within)
        initial_fits[(e,label)] = (b,s,sd,n)
        print(e, label, "I", f"{b[1]:+.4f} ({s[1]:.4f})",
              "NE", f"{b[2]:+.4f} ({s[2]:.4f})",
              "SD", round(sd,3), "n", n)

print("\nVocab numeric note:", VOCAB.get("numeric_note"))
print("Tuning IDs for first validation batch:")
print([int(x) for x in tuning.stay_id.drop_duplicates().iloc[:100]])
print("Counts:", RUNS, "Python;", FORWARDS, "forward.")
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 2\nTraining has 30,000 stays and 1,997,441 events. "
            "Measurements at integer hours; actions at hour+0.0833, lasting one hour. "
            "Insulin rates are quantized in 0.5 steps; norepinephrine rates in 0.01 "
            "steps and capped at 1. No duplicate code/stay/hour rows. "
            "Constructed hourly panel and began raw/history/within-stay comparisons.\n")
```