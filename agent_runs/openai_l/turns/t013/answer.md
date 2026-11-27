The fixed-effect dynamical model also disagrees with the threshold estimates, so I won’t use either state-space fit as causal truth. Insulin has substantial dose variation within stays. I’ll test dose innovations after flexible, cross-fitted history adjustment for both drugs and sensors. Stable estimates across sensor and history subsets would strengthen the evidence; instability would expose unresolved confounding.

```python
import numpy as np, time
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

# All features are available BEFORE the current hour's drug bundle.
# No future readings, future doses, or whole-stay averages enter these controls.
hf = {}
hh = np.tile(np.arange(24),len(rec_ids))
hf["hour"] = hh
for e in P:
    last = Last[e][:,:24]
    obs = np.isfinite(P[e][:,:24])
    hist_full = np.column_stack([pre[e].to_numpy(),P[e][:,:24]])
    positions = np.maximum.accumulate(
        np.where(np.isfinite(hist_full),np.arange(25),0),axis=1)[:,1:]
    hf[e+"_last"] = last.ravel()
    hf[e+"_observed"] = obs.ravel().astype(float)
    hf[e+"_age"] = (np.arange(24)[None,:]+1-positions).ravel()
    hf[e+"_initial"] = np.repeat(pre[e].to_numpy(),24)
    for lag in [1,2,3]:
        lagged = np.column_stack([np.repeat(pre[e].to_numpy()[:,None],lag,axis=1),
                                 last[:,:24-lag]])
        hf[e+"_lag"+str(lag)] = lagged.ravel()
    sums = np.nancumsum(hist_full,axis=1)[:,1:]
    counts = np.cumsum(np.isfinite(hist_full),axis=1)[:,1:]
    hf[e+"_past_mean"] = (sums/np.maximum(counts,1)).ravel()
for drug in A:
    for lag in [1,2,3]:
        hf[drug+"_lag"+str(lag)] = np.column_stack(
            [np.zeros((len(rec_ids),lag)),A[drug][:,:24-lag]]).ravel()
    hf[drug+"_past_total"] = np.column_stack(
        [np.zeros(len(rec_ids)),np.cumsum(A[drug][:,:23],axis=1)]).ravel()
hf["insulin_protocol_base"] = np.nan_to_num(ins_base,nan=0).ravel()
hf["NE_below65_last_arterial"] = np.maximum(0,65-Last["map_arterial"][:,:24]).ravel()
hf["NE_below65_last_cuff"] = np.maximum(0,65-Last["map_cuff"][:,:24]).ravel()
hist_names = list(hf)
Hhist = np.column_stack([hf[k] for k in hist_names]).astype(np.float32)
fold_hist = np.repeat(np.arange(len(rec_ids))%2,24)
group_hist = np.repeat(np.arange(len(rec_ids)),24)
rng_hist = np.random.default_rng(20261111)

def hist_fit_predict(y, fold, observed=None):
    if observed is None:
        observed = np.isfinite(y)
    tr = np.flatnonzero((fold_hist!=fold)&observed)
    if len(tr)>160000:
        tr = rng_hist.choice(tr,160000,replace=False)
    te = np.flatnonzero((fold_hist==fold)&observed)
    reg = HistGradientBoostingRegressor(
        max_iter=100,max_leaf_nodes=31,max_bins=128,learning_rate=.09,
        min_samples_leaf=80,l2_regularization=10,early_stopping=False)
    reg.fit(Hhist[tr],y[tr])
    return te,reg.predict(Hhist[te]),reg

def residual_effect(ra,ry,mask):
    sel = mask&np.isfinite(ra)&np.isfinite(ry)
    aa,yy = ra[sel],ry[sel]
    den = aa@aa
    beta = (aa@yy)/den
    scores = aa*(yy-beta*aa)
    sums = np.bincount(group_hist[sel],weights=scores,minlength=len(rec_ids))
    se = np.sqrt(sums@sums)/den
    return int(sel.sum()),float(beta),float(se),float(np.sqrt(den/sel.sum()))

dml_dose_resid = {}; dml_outcome_resid = {}; dml_models = {}
start = time.time()
with threadpool_limits(limits=8):
    for drug in ["insulin","norepinephrine"]:
        yy = A[drug][:,:24].ravel()
        pred = np.zeros(len(yy))
        for fold in [0,1]:
            te,pp,reg = hist_fit_predict(yy,fold)
            pred[te] = pp
            dml_models[(drug,fold)] = reg
        dml_dose_resid[drug] = yy-pred
        print("Policy",drug,"cross-fit R2",round(1-np.mean((yy-pred)**2)/np.var(yy),4),
              "residual SD",round(np.std(yy-pred),4))
    for target in P:
        yy = P[target][:,1:25].ravel()
        pred = np.full(len(yy),np.nan)
        for fold in [0,1]:
            te,pp,reg = hist_fit_predict(yy,fold)
            pred[te] = pp
            dml_models[(target,fold)] = reg
        dml_outcome_resid[target] = yy-pred
        drug = "insulin" if target.startswith("glucose") else "norepinephrine"
        paired = ["glucose_finger","glucose_lab"] if drug=="insulin" else ["map_arterial","map_cuff"]
        masks = {
            "all": np.ones(len(yy),dtype=bool),
            "active": A[drug][:,:24].ravel()>0,
            "both current": np.isfinite(P[paired[0]][:,:24]).ravel()&
                            np.isfinite(P[paired[1]][:,:24]).ravel()}
        for label,mask in masks.items():
            ans = residual_effect(dml_dose_resid[drug],dml_outcome_resid[target],mask)
            print("DML",target,label,"n/beta/clusterSE/doseResidualSD",
                  tuple(round(v,4) if isinstance(v,float) else v for v in ans))
print("Total seconds",round(time.time()-start,1),"features",Hhist.shape)

with open("/work/notes.md","a") as f:
    f.write("\n## Run 11\nProfiled fixed-effect SSM also gives incompatible effects (I -.419,rho .772; NE .435,rho .602), so neither simple SSM is accepted as a causal model. Insulin dose/base SD .343 overall and .330 within stay provides substantial within-stay variation. Built pre-action history features only and ran two-fold stay-separated flexible residualization (small statistical regressions, not world-model training), comparing all/active/both-current subsets and both sensors.\n")
```