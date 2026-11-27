Drugs are recorded five minutes after hourly readings, so the next hour’s reading follows the infusion. This gives a useful temporal ordering, but doses clearly may depend on current condition. I’ll compare dose associations before and after controlling for prior readings and stay-level differences; a stable residual-dose effect would support a causal estimate, while instability would warn of confounding.

```python
from scipy import linalg

measures = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drugs = ["insulin", "norepinephrine"]
tr = train.copy()
tr["t"] = np.floor(tr.hour + 1e-5).astype(int)
grid = pd.MultiIndex.from_product([ids, range(-1,25)], names=["stay_id","t"])
wide = tr[tr.event.isin(measures)].pivot(
    index=["stay_id","t"], columns="event", values="value"
).reindex(grid)
rates = tr[tr.event.isin(drugs)].pivot(
    index=["stay_id","t"], columns="event", values="rate"
).reindex(grid).fillna(0)
wide = wide.join(rates)
for col in measures + drugs:
    wide[col+"_prev"] = wide.groupby(level=0)[col].shift(1)
for col in measures:
    wide[col+"_next"] = wide.groupby(level=0)[col].shift(-1)
for col in measures:
    baseline = wide[col].unstack("t")[[-1,0]].mean(axis=1)
    wide[col+"_baseline"] = baseline.reindex(wide.index.get_level_values(0)).to_numpy()
wide["hour"] = wide.index.get_level_values("t").astype(float)

print("Reading distributions:\n", wide[measures].describe().round(3).to_string())
for a,b in [("glucose_finger","glucose_lab"),("map_arterial","map_cuff")]:
    pair = wide[[a,b]].dropna()
    diff = pair[a]-pair[b]
    print("Paired", a, b, "n", len(pair), "corr", round(pair.corr().iloc[0,1],4),
          "difference quantiles", diff.quantile([0,.1,.5,.9,1]).to_dict())

def clustered_ols(X, y, groups):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    coef = linalg.lstsq(X,y,lapack_driver="gelsy")[0]
    resid = y-X@coef
    inv = np.linalg.pinv(X.T@X, rcond=1e-12)
    _, gi = np.unique(groups, return_inverse=True)
    scores = np.zeros((gi.max()+1, X.shape[1]))
    np.add.at(scores, gi, X*resid[:,None])
    se = np.sqrt(np.maximum(np.diag(inv@(scores.T@scores)@inv),0))
    return coef, se, resid

def initial_effect(outcome, drug, counterpart, fixed_effect=False, positive_only=False):
    q = wide[(wide.hour>=1)&(wide.hour<=23)].copy()
    q = q.dropna(subset=[outcome,outcome+"_next"])
    if positive_only:
        q = q[q[drug]>0]
    scale = 100. if "glucose" in outcome else 50.
    v = q[outcome].to_numpy()/scale
    cols = {"intercept":np.ones(len(q)), "dose":q[drug].to_numpy(),
            "present":(q[drug]>0).astype(float).to_numpy(),
            "current":v, "current2":v*v, "current3":v*v*v,
            "hour":q.hour.to_numpy()/24}
    for c in [outcome+"_prev", counterpart, counterpart+"_prev", outcome+"_baseline"]:
        x = q[c]
        cols[c] = x.fillna(x.median()).to_numpy()/scale
        cols[c+"_missing"] = x.isna().astype(float).to_numpy()
    cols["dose_prev"] = q[drug+"_prev"].fillna(0).to_numpy()
    labels = list(cols)
    X = np.column_stack(list(cols.values()))
    y = (q[outcome+"_next"]-q[outcome]).to_numpy()
    groups = q.index.get_level_values("stay_id").to_numpy()
    if fixed_effect:
        _,gi = np.unique(groups,return_inverse=True)
        n = np.bincount(gi)
        sx = np.zeros((len(n),X.shape[1])); np.add.at(sx,gi,X)
        sy = np.bincount(gi,weights=y)
        X -= (sx/n[:,None])[gi]
        y -= (sy/n)[gi]
    # Remove redundant constant columns before inference.
    keep = np.std(X,axis=0)>1e-9
    if not fixed_effect: keep[0]=True
    X=X[:,keep]; labels=[s for s,k in zip(labels,keep) if k]
    b,se,res=clustered_ols(X,y,groups)
    j=labels.index("dose")
    rest=np.delete(X,j,axis=1)
    dose_res=X[:,j]-rest@linalg.lstsq(rest,X[:,j],lapack_driver="gelsy")[0]
    print(outcome,drug,"FE",fixed_effect,"positive",positive_only,"n",len(q),
          "dose effect",round(b[j],5),"+/-",round(1.96*se[j],5),
          "residual dose SD",round(dose_res.std(),5),
          "outcome residual SD",round(res.std(),3))
    return {"coef":b[j],"se":se[j],"n":len(q),"dose_res_sd":dose_res.std()}

initial_results = []
for outcome,drug,counterpart in [
    ("glucose_finger","insulin","glucose_lab"),
    ("map_arterial","norepinephrine","map_cuff")]:
    for fe in [False,True]:
        for pos in [False,True]:
            initial_results.append(initial_effect(outcome,drug,counterpart,fe,pos))

print("\nDose versus current reading, drug-present observations:")
for outcome,drug in [("glucose_finger","insulin"),("map_arterial","norepinephrine")]:
    q=wide[(wide[drug]>0)&wide[outcome].notna()].copy()
    q["reading_decile"]=pd.qcut(q[outcome],10,duplicates="drop")
    print(q.groupby("reading_decile",observed=True).agg(
        n=(drug,"size"),reading=(outcome,"mean"),rate=(drug,"mean"),
        rate_sd=(drug,"std")).round(3).to_string())

with open("/work/notes.md","a") as f:
    f.write("\n## Run 2\n30,000 stays, 2,061,968 events. Readings occur at "
            "integer hours; one-hour drug records start at hour + 0.0833. "
            "Rates are discrete/rounded; insulin mean positive rate 2.198, "
            "norepinephrine 0.537. No causal conclusions yet.\n")
    f.write("\n## Run 3: initial regressions\n"+repr(initial_results)+"\n")
```
