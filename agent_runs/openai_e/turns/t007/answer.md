The insulin discontinuities persist for several hours; a simple dynamic fit gives an immediate effect near −2.28 and persistence near 0.86. That suggests a larger cumulative rollout effect than the original model predicts. Norepinephrine dosing appears strongly history-dependent, so I’ll use flexible history adjustment and inspect its dosing rule. Stable residual-dose estimates would strengthen identification.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
import time

def nuisance_features(s, nlags=4):
    columns=[s.t.to_numpy()/24]
    labels=["hour"]
    for c in MEAS:
        for suffix in ["", "_last", "_age", "_base"]:
            columns.append(s[c+suffix].to_numpy())
            labels.append(c+suffix)
        for lag in range(1,nlags+1):
            columns.append(s[c+f"_lag{lag}"].to_numpy())
            labels.append(c+f"_lag{lag}")
    for c in DRUGS:
        for lag in range(1,nlags+1):
            columns.append(s[c+f"_lag{lag}"].to_numpy())
            labels.append(c+f"_lag{lag}")
    return np.column_stack(columns).astype(np.float32), labels

sne=panel[(panel.t>=4)&(panel.t<24)&panel.map_arterial.notna()&
          panel.map_arterial_next.notna()].copy().reset_index(drop=True)
Xne,feature_names=nuisance_features(sne,4)
yne=sne.map_arterial_next.to_numpy()
dne=sne.norepinephrine.to_numpy()
fold=(sne.stay_id.to_numpy()%2).astype(int)
pred_y=np.zeros(len(sne))
pred_d=np.zeros(len(sne))
ne_nuisance=[]
start=time.time()
for f in [0,1]:
    tr=(fold!=f); te=~tr
    models=[]
    for target,pred in [(yne,pred_y),(dne,pred_d)]:
        model=HistGradientBoostingRegressor(
            max_iter=160,max_leaf_nodes=31,max_depth=None,
            learning_rate=.08,l2_regularization=10,
            min_samples_leaf=100,max_bins=128,
            early_stopping=False,random_state=912+f)
        model.fit(Xne[tr],target[tr])
        pred[te]=model.predict(Xne[te])
        models.append(model)
    ne_nuisance.append(models)
print("Nuisance fit seconds:",round(time.time()-start,1))
rd=dne-pred_d
ry=yne-pred_y
b,se,sd=ols_cluster(np.column_stack([np.ones(len(sne)),rd]),
                    ry,sne.stay_id.to_numpy())
NE_DML=float(b[-1])
print("=== CROSS-FITTED NE EFFECT ===")
print("n",len(sne),"beta",round(NE_DML,5),"cluster_SE",round(se[-1],5),
      "dose_residual_sd",round(rd.std(),5),
      "outcome_RMSE",round(np.sqrt(np.mean(ry**2)),4),
      "dose_RMSE",round(np.sqrt(np.mean(rd**2)),5))
for lo,hi in [(0,.15),(.15,.5),(.5,1.1)]:
    keep=(pred_d>=lo)&(pred_d<hi)
    bb,ss,_=ols_cluster(np.column_stack([np.ones(keep.sum()),rd[keep]]),
                        ry[keep],sne.stay_id.to_numpy()[keep])
    print("Expected-dose band",lo,hi,"n",int(keep.sum()),
          "beta",round(bb[-1],4),"SE",round(ss[-1],4))

print("\n=== NE FEEDBACK-RULE DIAGNOSTIC ===")
prev=sne.norepinephrine_lag1.to_numpy()
m=sne.map_arterial.to_numpy()
interior=(dne>.02)&(dne<.95)&(prev>.03)&(prev<.8)
rule_rows=[]
for gain in [.005,.008,.01,.012,.015,.02]:
    base=np.maximum(prev+gain*(65-m),.01)
    z=np.log(dne[interior]/base[interior])
    rule_rows.append([gain,z.mean(),z.std(),np.corrcoef(z,m[interior])[0,1],
                      np.corrcoef(z,prev[interior])[0,1]])
print(pd.DataFrame(rule_rows,columns=["feedback_gain","log_ratio_mean",
                                     "log_ratio_sd","corr_MAP","corr_prev_rate"]
                   ).round(4).to_string(index=False))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 5–6: dynamic identification\n"
            "Pooled insulin RD excludes rounded cutoff values at 180/220. "
            "Immediate glucose jump -2.2131 (SE .1583), dose jump 1.0361. "
            "Reduced-form outcome discontinuity fades over six hours; downstream dosing also changes. "
            "Joint impulse fit beta=-2.2778, persistence=.85743; provisional, not exact truth. "
            "NE polynomial-history coefficients remain about +3.09, but dose-policy nonlinearity may bias them. "
            "Cross-fitted flexible nuisance models now test residual-dose effects.\n"
            f"NE DML beta={NE_DML:.6f}, cluster SE={se[-1]:.6f}.\n"
            "Python runs: 6. Forward requests: 1.\n")
```
