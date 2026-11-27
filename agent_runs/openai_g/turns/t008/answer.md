Insulin remains near zero through 12 hours. Norepinephrine shows a positive 12-hour association (+0.83 per rate unit, SE 0.23), so treating it as zero would be premature. I’ll add a much fuller pre-dose history and check cuff readings too. Persistence across these checks would support a delayed effect; disappearance would favor omitted-history or recording-pattern bias.

```python
full_features=features.copy()
for c in measures+drugs:
    for lag in range(2,11):
        full_features[c+"_lag"+str(lag)] = wide[c].groupby(level=0).shift(lag)
for c in measures:
    for lag in [1,2,4,8]:
        full_features[c+"_last_lag"+str(lag)] = (
            features[c+"_last"].groupby(level=0).shift(lag))
    v=wide[c].to_numpy(dtype=float).reshape(len(ids),26)
    sums=np.cumsum(np.nan_to_num(v,nan=0),axis=1)
    counts=np.cumsum(np.isfinite(v),axis=1)
    for window in [3,6,12]:
        old_sum=np.zeros_like(sums); old_count=np.zeros_like(counts)
        old_sum[:,window:]=sums[:,:-window]
        old_count[:,window:]=counts[:,:-window]
        s=sums-old_sum; n=counts-old_count
        mean=np.divide(s,n,out=np.full_like(s,np.nan),where=n>0)
        full_features[c+"_mean"+str(window)]=mean.ravel()
        full_features[c+"_count"+str(window)]=n.ravel()
for c in drugs:
    v=wide[c+"_prev"].fillna(0).to_numpy().reshape(len(ids),26)
    sums=np.cumsum(v,axis=1)
    for window in [3,6,12]:
        old=np.zeros_like(sums); old[:,window:]=sums[:,:-window]
        full_features[c+"_sum_prev"+str(window)]=(sums-old).ravel()

drug="norepinephrine"
decision=(wide.hour>=0)&(wide.hour<=23)
q0=wide.loc[decision]
Xf=full_features.loc[decision].copy()
Xf["drug_present"]=(q0[drug]>0).astype(float)
Xf=np.asarray(Xf,dtype=np.float32)
A0=q0[drug].to_numpy(dtype=np.float64)
groups0=q0.index.get_level_values("stay_id").to_numpy()
folds0=(groups0%2).astype(int)
pa0=np.zeros(len(q0))
for fold in [0,1]:
    fit=folds0!=fold; test=~fit
    m=HistGradientBoostingRegressor(
        max_iter=350,max_leaf_nodes=31,min_samples_leaf=150,
        learning_rate=.06,l2_regularization=2.,random_state=420+fold
    ).fit(Xf[fit],A0[fit])
    pa0[test]=m.predict(Xf[test])
raf=A0-pa0
print("Expanded history feature count:",Xf.shape[1],
      "dose residual SD:",round(raf.std(),5),
      "old residual SD:",round(dose_innovations[drug].std(),5))
extended_results=[]
for outcome,horizon in [("map_arterial",1),("map_arterial",4),
                        ("map_arterial",8),("map_arterial",12),
                        ("map_cuff",8),("map_cuff",12)]:
    future=wide[outcome].groupby(level=0).shift(-horizon).loc[decision]
    valid=future.notna().to_numpy() & (q0.hour.to_numpy()<=24-horizon)
    X=Xf[valid]; Y=future.to_numpy(dtype=float)[valid]
    a=raf[valid]; A=A0[valid]
    groups=groups0[valid]; folds=folds0[valid]
    py=np.zeros(len(Y))
    for fold in [0,1]:
        fit=folds!=fold; test=~fit
        m=HistGradientBoostingRegressor(
            max_iter=250,max_leaf_nodes=31,min_samples_leaf=150,
            learning_rate=.06,l2_regularization=2.,random_state=500+horizon+fold
        ).fit(X[fit],Y[fit])
        py[test]=m.predict(X[test])
    ry=Y-py
    for label,sub in [("positive",A>0),("positive_unsaturated",(A>0)&(A<.95))]:
        aa=a[sub]; yy=ry[sub]; gg=groups[sub]
        b=np.dot(aa,yy)/np.dot(aa,aa)
        _,gi=np.unique(gg,return_inverse=True)
        score=np.bincount(gi,weights=aa*(yy-b*aa))
        se=np.sqrt(np.dot(score,score))/np.dot(aa,aa)
        r={"outcome":outcome,"horizon":horizon,"subset":label,
           "n":int(sub.sum()),"coef":float(b),"se":float(se)}
        extended_results.append(r)
        print(r)

print("\nTwo norepinephrine trajectories:")
ne_ids=wide[wide.norepinephrine>0].groupby(level=0).size().sort_values(
    ascending=False).index[:2]
for sid in ne_ids:
    print("Stay",sid)
    print(wide.loc[sid,["map_arterial","map_cuff","norepinephrine"]].round(3).to_string())

with open("/work/notes.md","a") as f:
    f.write("\n## Expanded-history NE checks\n"+repr(extended_results)+"\n")
    f.write("Initial long-horizon NE association was +0.832 (SE 0.231) at "
            "12h; this contradicted a blanket null and motivated richer "
            "history controls and alternate-measurement replication. No edits yet.\n")
```
