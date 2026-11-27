Insulin dose thresholds coincide with lower next-hour glucose, suggesting a real negative effect. However, readings are rounded and monitoring frequency also changes at the first threshold, so the initial estimates need refinement. I’ll exclude ambiguous threshold values and independently estimate effects using dose variation left unexplained by pre-dose history. Agreement between these methods would strengthen the conclusion.

```python
PY_RUNS += 1
from sklearn.ensemble import HistGradientBoostingRegressor
import time

rd_refined = []
for cutoff in [150.,180.,220.,260.]:
    for bw in [5,10,15,20]:
        elig = valid_hours & fresh_g & (policy_g.to_numpy()!=cutoff)
        rd_refined.append(local_rd(policy_g,pan.insulin,next_obs.glucose_finger,
                                   cutoff,bw,elig))
rd_refined = pd.DataFrame(rd_refined)
print("=== RD EXCLUDING ROUNDED THRESHOLD VALUES ===")
print(rd_refined.round(4).to_string(index=False))

def build_history_features(p):
    h = p.index.get_level_values("t").to_numpy().astype(float)
    obs = p[measure_events]
    ll = obs.groupby(level=0).ffill()
    parts = [pd.DataFrame({"hour":h}, index=p.index),
             ll.add_prefix("last_"),
             obs.notna().astype(float).add_prefix("seen_now_")]
    for lag in [1,2,3]:
        parts.append(ll.groupby(level=0).shift(lag).add_prefix(f"last_lag{lag}_"))
    seen_time = pd.DataFrame(
        np.where(obs.notna(),h[:,None],np.nan), index=p.index, columns=measure_events)
    last_time = seen_time.groupby(level=0).ffill()
    ages = pd.DataFrame(h[:,None]-last_time.to_numpy(),index=p.index,
                        columns=[f"age_{c}" for c in measure_events])
    parts.append(ages)
    counts = obs.notna().astype(float).groupby(level=0).cumsum()
    sums = obs.fillna(0).groupby(level=0).cumsum()
    parts.append((sums/counts.replace(0,np.nan)).add_prefix("past_mean_"))
    for lag in [1,2,3]:
        parts.append(p[drug_events].groupby(level=0).shift(lag).add_prefix(f"dose_lag{lag}_"))
    parts.append(p[drug_events].groupby(level=0).cumsum().groupby(level=0)
                 .shift(1).add_prefix("prior_cumulative_"))
    pg = ffill_stay(p.glucose_finger.combine_first(p.glucose_lab))
    pm = ffill_stay(p.map_arterial.combine_first(p.map_cuff))
    parts.append(pd.DataFrame({"policy_glucose":pg,"policy_map":pm},index=p.index))
    return pd.concat(parts,axis=1).astype(np.float32)

HX = build_history_features(pan)
fold = stay_codes % 2
dml_results = []
dml_residuals = {}
dml_models = {}
start = time.time()

for drug, target in [("insulin","glucose_finger"),("norepinephrine","map_arterial")]:
    ok = valid_hours & next_obs[target].notna().to_numpy()
    X = HX.loc[ok].to_numpy()
    D = pan.loc[ok,drug].to_numpy()
    Y = (next_obs[target]-last[target]).loc[ok].to_numpy()
    good = np.isfinite(Y)
    inds = np.flatnonzero(ok)[good]
    X,D,Y = X[good],D[good],Y[good]
    ff = fold[inds]
    predD = np.empty(len(D)); predY = np.empty(len(Y))
    models = []
    for f in [0,1]:
        fit = ff != f
        test = ff == f
        kwargs = dict(max_iter=120,max_leaf_nodes=24,min_samples_leaf=150,
                      learning_rate=.08,l2_regularization=10,max_bins=127,
                      early_stopping=False,random_state=1729+f)
        md = HistGradientBoostingRegressor(**kwargs).fit(X[fit],D[fit])
        my = HistGradientBoostingRegressor(**kwargs).fit(X[fit],Y[fit])
        predD[test] = md.predict(X[test])
        predY[test] = my.predict(X[test])
        models.append((md,my))
    rd = D-predD; ry = Y-predY
    beta = np.dot(rd,ry)/np.dot(rd,rd)
    influence = rd*(ry-beta*rd)/np.dot(rd,rd)
    sums = np.bincount(stay_codes[inds],weights=influence,minlength=len(train_ids))
    se = np.sqrt(np.dot(sums,sums))
    row = {"drug":drug,"target":target,"n":len(D),"beta":beta,
           "cluster_se":se,"dose_R2":1-np.mean(rd**2)/np.var(D),
           "dose_residual_SD":np.std(rd),"outcome_residual_SD":np.std(ry)}
    dml_results.append(row)
    dml_residuals[drug] = {"indices":inds,"dose":D,"outcome_change":Y,
                         "dose_prediction":predD,"outcome_prediction":predY,
                         "rd":rd,"ry":ry}
    dml_models[drug] = models
    print("\nCompleted:",row,"elapsed_s",round(time.time()-start,1))

dml_results = pd.DataFrame(dml_results)
print("\n=== CROSS-FITTED HISTORY-ADJUSTED EFFECTS ===")
print(dml_results.round(5).to_string(index=False))
print("History feature count:",HX.shape[1])
print("Elapsed seconds:",round(time.time()-start,1))

note(f"""## Run {PY_RUNS}
RD at the initial insulin threshold was complicated by rounded readings and
a discontinuous next-reading probability (about 30% to 100%).
Refined RD excludes the rounded threshold itself.
Refined RD:\n{rd_refined.round(4).to_string(index=False)}
Two-fold, stay-separated history-adjusted dose/outcome residual regressions:
{dml_results.round(5).to_string(index=False)}
These estimates require dose innovations to be unconfounded conditional on
recorded history; threshold agreement and placebo tests are used as checks.
No weights edited.
""")
```
