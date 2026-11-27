The original rollouts are already weak: insulin doubling/halving shifts glucose by +0.113/−0.186, and norepinephrine shifts MAP by +0.050/+0.013. Those are predictions, not evidence of drug effects. I’ll now use nonlinear, stay-separated adjustment and insulin dose-threshold checks. Consistent near-zero results would support making dose amount invariant; a reproducible nonzero result would refute that approach.

```python
import os, time
os.environ["OMP_NUM_THREADS"]="8"
from sklearn.ensemble import HistGradientBoostingRegressor

features = pd.DataFrame(index=wide.index)
features["hour"] = wide["hour"]
for c in measures:
    features[c] = wide[c]
    features[c+"_prev"] = wide[c+"_prev"]
    features[c+"_baseline"] = wide[c+"_baseline"]
    features[c+"_last"] = wide[c].groupby(level=0).ffill()
    observed_hour = wide["hour"].where(wide[c].notna()).groupby(level=0).ffill()
    features[c+"_age"] = wide["hour"]-observed_hour
    features[c+"_observed"] = wide[c].notna().astype(float)
for c in drugs:
    features[c+"_prev"] = wide[c+"_prev"].fillna(0)
    features[c+"_prev2"] = wide[c+"_prev"].groupby(level=0).shift(1).fillna(0)

dml_results=[]
for outcome,drug in [("glucose_finger","insulin"),("map_arterial","norepinephrine")]:
    start=time.time()
    mask=(wide.hour>=0)&(wide.hour<=23)&wide[outcome+"_next"].notna()
    q=wide.loc[mask]
    X=features.loc[mask].copy()
    X["drug_present"]=(q[drug]>0).astype(float)
    X=np.asarray(X,dtype=np.float32)
    A=q[drug].to_numpy(dtype=np.float64)
    Y=q[outcome+"_next"].to_numpy(dtype=np.float64)
    groups=q.index.get_level_values("stay_id").to_numpy()
    folds=(groups % 2).astype(int)
    pa=np.zeros(len(q)); py=np.zeros(len(q))
    for fold in [0,1]:
        fit=folds!=fold; test=~fit
        kwargs=dict(max_iter=100,max_leaf_nodes=31,min_samples_leaf=100,
                    learning_rate=.08,l2_regularization=1.,random_state=120+fold)
        ma=HistGradientBoostingRegressor(**kwargs).fit(X[fit],A[fit])
        my=HistGradientBoostingRegressor(**kwargs).fit(X[fit],Y[fit])
        pa[test]=ma.predict(X[test]); py[test]=my.predict(X[test])
    ra=A-pa; ry=Y-py
    for label,sub in [
        ("all",np.ones(len(q),dtype=bool)),
        ("positive",A>0),
        ("current_observed",q[outcome].notna().to_numpy()),
        ("positive_unsaturated",(A>0)&((A<.95) if drug=="norepinephrine" else (A<8)))
    ]:
        a=ra[sub]; y=ry[sub]; g=groups[sub]
        b=np.dot(a,y)/np.dot(a,a)
        _,gi=np.unique(g,return_inverse=True)
        sc=np.bincount(gi,weights=a*(y-b*a))
        se=np.sqrt(np.dot(sc,sc))/np.dot(a,a)
        r={"outcome":outcome,"drug":drug,"subset":label,"n":int(sub.sum()),
           "coef":float(b),"se":float(se),"dose_res_sd":float(a.std())}
        dml_results.append(r)
        print("DML",r)
    print("Seconds:",round(time.time()-start,1))

print("\nInsulin local dose-threshold checks (exclude the exact threshold):")
rd_results=[]
for cutoff in [150,180,220,260,300]:
    q=wide[(wide.hour>=0)&(wide.hour<=23)].dropna(
        subset=["glucose_finger","glucose_finger_next"]).copy()
    x=q.glucose_finger.to_numpy()-cutoff
    sel=(np.abs(x)<=12)&(x!=0)
    q=q.loc[sel]; x=x[sel]/12
    right=(x>0).astype(float)
    base=q.glucose_finger_baseline.fillna(cutoff).to_numpy()/100
    prev=q.glucose_finger_prev.fillna(q.glucose_finger).to_numpy()/100
    X=np.column_stack([np.ones(len(q)),right,x,x*right,
                       q.hour.to_numpy()/24,base,prev])
    groups=q.index.get_level_values("stay_id").to_numpy()
    A=q.insulin.to_numpy()
    Y=(q.glucose_finger_next-q.glucose_finger).to_numpy()
    ba,sa,_=clustered_ols(X,A,groups)
    by,sy,_=clustered_ols(X,Y,groups)
    fs=ba[1]; rf=by[1]
    effect=rf/fs if abs(fs)>.05 else np.nan
    _,ratio_se,_=clustered_ols(X,Y-effect*A,groups)
    r={"cutoff":cutoff,"n":len(q),"dose_jump":float(fs),
       "outcome_jump":float(rf),"effect_per_unit":float(effect),
       "se":float(ratio_se[1]/abs(fs))}
    rd_results.append(r)
    print("RD",r)

with open("/work/notes.md","a") as f:
    f.write("\n## Forward 1: original tuning stays 0–99\n"
            "Insulin x2 +0.113, half -0.186; NE x2 +0.050, half +0.013. "
            "These are model responses, not true effects.\n")
    f.write("\n## Nonlinear cross-fitted and threshold checks\n"+
            repr(dml_results)+"\n"+repr(rd_results)+"\n")
```
