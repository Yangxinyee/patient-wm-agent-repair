The first two adjusted insulin estimates are near zero despite substantial residual dose variation. That supports a negligible insulin effect, but it is still provisional: the stay-fixed-effect check failed on a read-only array. I’ll fix that implementation issue and finish both drugs’ estimates; agreement across specifications would strengthen the conclusion.

```python
def initial_effect(outcome, drug, counterpart, fixed_effect=False, positive_only=False):
    q = wide[(wide.hour>=1)&(wide.hour<=23)].dropna(
        subset=[outcome,outcome+"_next"]).copy()
    if positive_only:
        q = q[q[drug]>0]
    scale = 100. if "glucose" in outcome else 50.
    v = q[outcome].to_numpy()/scale
    cols = {"intercept":np.ones(len(q)), "dose":q[drug].to_numpy(),
            "present":(q[drug]>0).astype(float).to_numpy(),
            "current":v, "current2":v*v, "current3":v*v*v,
            "hour":q.hour.to_numpy()/24}
    for c in [outcome+"_prev", counterpart, counterpart+"_prev", outcome+"_baseline"]:
        x=q[c]
        cols[c]=x.fillna(x.median()).to_numpy()/scale
        cols[c+"_missing"]=x.isna().astype(float).to_numpy()
    cols["dose_prev"]=q[drug+"_prev"].fillna(0).to_numpy()
    labels=list(cols)
    X=np.column_stack(list(cols.values())).astype(np.float64)
    y=np.array(q[outcome+"_next"]-q[outcome], dtype=np.float64, copy=True)
    groups=q.index.get_level_values("stay_id").to_numpy()
    if fixed_effect:
        _,gi=np.unique(groups,return_inverse=True)
        n=np.bincount(gi)
        sx=np.zeros((len(n),X.shape[1])); np.add.at(sx,gi,X)
        sy=np.bincount(gi,weights=y)
        X-=(sx/n[:,None])[gi]
        y-=(sy/n)[gi]
    keep=np.std(X,axis=0)>1e-9
    if not fixed_effect: keep[0]=True
    X=X[:,keep]; labels=[s for s,k in zip(labels,keep) if k]
    b,se,res=clustered_ols(X,y,groups)
    j=labels.index("dose")
    rest=np.delete(X,j,axis=1)
    dose_res=X[:,j]-rest@linalg.lstsq(rest,X[:,j],lapack_driver="gelsy")[0]
    result={"outcome":outcome,"drug":drug,"FE":fixed_effect,"positive":positive_only,
            "n":len(q),"coef":float(b[j]),"se":float(se[j]),
            "dose_res_sd":float(dose_res.std()),"resid_sd":float(res.std())}
    print(result)
    return result

initial_results=[]
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
    print(outcome,drug)
    print(q.groupby("reading_decile",observed=True).agg(
        n=(drug,"size"),reading=(outcome,"mean"),rate=(drug,"mean"),
        rate_sd=(drug,"std")).round(3).to_string())

with open("/work/notes.md","a") as f:
    f.write("\n## Runs 3–4: initial causal checks\n"
            "Run 3 interrupted by a pandas read-only array during FE demeaning; "
            "fixed with an explicit copy. These are observational regressions, "
            "not yet definitive causal estimates.\n"+repr(initial_results)+"\n")
```
