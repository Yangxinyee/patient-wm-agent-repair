Both nonlinear dose estimates are near zero, and insulin’s dose jumps at five thresholds produce no detectable next-hour glucose jump. This strengthens the negligible-effect hypothesis and explains the earlier norepinephrine discrepancy as likely adjustment bias. Before editing, I’ll check longer horizons and alternate measurement types; a delayed response would rule out simply removing dose sensitivity.

```python
lag_results=[]
dose_innovations={}
for target,drug,alternate in [
    ("glucose_finger","insulin","glucose_lab"),
    ("map_arterial","norepinephrine","map_cuff")
]:
    decision=(wide.hour>=0)&(wide.hour<=23)
    q0=wide.loc[decision]
    X0=features.loc[decision].copy()
    X0["drug_present"]=(q0[drug]>0).astype(float)
    X0=np.asarray(X0,dtype=np.float32)
    A0=q0[drug].to_numpy(dtype=np.float64)
    groups0=q0.index.get_level_values("stay_id").to_numpy()
    folds0=(groups0%2).astype(int)
    pa0=np.zeros(len(q0))
    for fold in [0,1]:
        fit=folds0!=fold; test=~fit
        m=HistGradientBoostingRegressor(
            max_iter=120,max_leaf_nodes=31,min_samples_leaf=100,
            learning_rate=.08,l2_regularization=1.,random_state=200+fold
        ).fit(X0[fit],A0[fit])
        pa0[test]=m.predict(X0[test])
    ra0=A0-pa0
    dose_innovations[drug]=pd.Series(ra0,index=q0.index)
    for outcome,horizon in [(target,h) for h in [2,4,8,12]]+[(alternate,1),(alternate,4)]:
        future=wide[outcome].groupby(level=0).shift(-horizon).loc[decision]
        valid=future.notna().to_numpy() & (q0.hour.to_numpy()<=24-horizon)
        X=X0[valid]; Y=future.to_numpy(dtype=np.float64)[valid]
        ra=ra0[valid]; A=A0[valid]
        groups=groups0[valid]; folds=folds0[valid]
        py=np.zeros(len(Y))
        for fold in [0,1]:
            fit=folds!=fold; test=~fit
            m=HistGradientBoostingRegressor(
                max_iter=100,max_leaf_nodes=31,min_samples_leaf=100,
                learning_rate=.08,l2_regularization=1.,random_state=300+horizon+fold
            ).fit(X[fit],Y[fit])
            py[test]=m.predict(X[test])
        ry=Y-py
        for label,sub in [("all",np.ones(len(Y),dtype=bool)),("positive",A>0)]:
            a=ra[sub]; y=ry[sub]; g=groups[sub]
            b=np.dot(a,y)/np.dot(a,a)
            _,gi=np.unique(g,return_inverse=True)
            score=np.bincount(gi,weights=a*(y-b*a))
            se=np.sqrt(np.dot(score,score))/np.dot(a,a)
            r={"drug":drug,"outcome":outcome,"horizon":horizon,"subset":label,
               "n":int(sub.sum()),"coef":round(float(b),5),"se":round(float(se),5)}
            lag_results.append(r)
            print(r)

print("\nDose-embedding tensor types and norms:")
for name in ["embed.numeric.weight","embed.bins.dose.weight",
             "embed.bins.rate.weight","embed.code.weight"]:
    a=load_original(name)
    print(name,type(a).__name__,a.shape,
          "norm",round(float(np.linalg.norm(np.asarray(a))),4))
print("Current edits:",edited())

with open("/work/notes.md","a") as f:
    f.write("\n## Delayed and alternate-outcome checks\n"+repr(lag_results)+"\n")
    f.write("No edits yet. Nonlinear one-hour effects: insulin +0.0466 "
            "(SE 0.0370); NE -0.0123 (SE 0.0819). Insulin threshold checks "
            "show strong dose jumps but no significant outcome jumps.\n")
```
