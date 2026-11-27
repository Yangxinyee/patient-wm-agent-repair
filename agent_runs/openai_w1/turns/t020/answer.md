The examples show doses starting about five minutes after the hourly readings, which gives a useful pre-dose history for adjustment. I’ll estimate dose-innovation responses over six hours, then separate persistence from later dose changes. Stable one-hour effects and coherent longer-lag responses would support a proportional repair; unstable estimates would argue against it.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import time
repair_start = time.time()
repair_events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
repair_drugs = ["insulin", "norepinephrine"]
repair_rates = {d: np.nan_to_num(np.asarray(R[d], dtype=float), nan=0.0) for d in repair_drugs}
repair_last = {}
repair_age = {}
repair_means = {}
for e in repair_events:
    aa = np.asarray(P[e], dtype=float)
    good = np.isfinite(aa)
    lastix = np.maximum.accumulate(np.where(good, np.arange(26)[None, :], -1), axis=1)
    repair_last[e] = np.take_along_axis(aa, np.maximum(lastix, 0), axis=1)
    repair_last[e][lastix < 0] = np.nan
    repair_age[e] = np.arange(26)[None, :] - lastix
    repair_means[e] = np.cumsum(np.nan_to_num(aa), axis=1) / np.maximum(np.cumsum(good, axis=1), 1)

repair_s, repair_j = np.indices((len(stay_ids), 16))
repair_s = repair_s.ravel()
repair_j = (repair_j + 5).ravel()  # exposure hours 3..18; index = hour + 1
repair_features = [repair_j.astype(float) - 1]
repair_feature_names = ["hour"]
for e in repair_events:
    for lag in range(4):
        repair_features.append(repair_last[e][repair_s, repair_j-lag])
        repair_feature_names.append(e + "_last_lag" + str(lag))
    repair_features += [
        repair_age[e][repair_s, repair_j],
        np.isfinite(P[e][repair_s, repair_j]).astype(float),
        repair_means[e][repair_s, repair_j],
        P[e][repair_s, 0], P[e][repair_s, 1]
    ]
    repair_feature_names += [e + suffix for suffix in ["_age", "_present", "_mean", "_initial", "_hour0"]]
for d in repair_drugs:
    for lag in range(1, 5):
        repair_features.append(repair_rates[d][repair_s, repair_j-lag])
        repair_feature_names.append(d + "_lag" + str(lag))
    repair_features.append(np.mean(repair_rates[d][:, :5], axis=1)[repair_s])
    repair_feature_names.append(d + "_earlymean")
repair_C = np.column_stack(repair_features).astype(np.float32)
repair_fold = ((stay_ids[repair_s] * 2654435761) % 2).astype(int)
repair_lp = {}
repair_hgb_params = dict(max_iter=90, max_leaf_nodes=15, learning_rate=.1,
                         min_samples_leaf=100, l2_regularization=10,
                         max_bins=63, early_stopping=False, random_state=117)

def repair_crossfit(target, mask, features, folds):
    pred = np.full(len(target), np.nan)
    for fold in (0, 1):
        tr = mask & (folds != fold)
        te = mask & (folds == fold)
        reg = HistGradientBoostingRegressor(**repair_hgb_params)
        reg.fit(features[tr], target[tr])
        pred[te] = reg.predict(features[te])
    return pred

def repair_cluster_beta(rd, ry, si):
    den = np.dot(rd, rd)
    beta = np.dot(rd, ry) / den
    sums = np.bincount(si, weights=rd * (ry-beta*rd), minlength=len(stay_ids))
    return float(beta), float(np.sqrt(np.dot(sums, sums)) / den)

with threadpool_limits(limits=8):
    for e, d in [("glucose_finger", "insulin"), ("map_arterial", "norepinephrine")]:
        mask0 = np.isfinite(P[e][repair_s, repair_j])
        target_d = repair_rates[d][repair_s, repair_j]
        md0 = repair_crossfit(target_d, mask0, repair_C, repair_fold)
        rd0 = target_d - md0
        y_resp, d_resp = [], [(1.0, 0.0)]
        for k in range(1, 7):
            yy = P[e][repair_s, repair_j+k]
            mask = mask0 & np.isfinite(yy)
            my = repair_crossfit(yy, mask, repair_C, repair_fold)
            y_resp.append(repair_cluster_beta(rd0[mask], (yy-my)[mask], repair_s[mask]))
            if k <= 5:
                dd = repair_rates[d][repair_s, repair_j+k]
                mf = repair_crossfit(dd, mask0, repair_C, repair_fold)
                d_resp.append(repair_cluster_beta(rd0[mask0], (dd-mf)[mask0], repair_s[mask0]))
        kernel = []
        for k in range(6):
            kernel.append(y_resp[k][0] - sum(kernel[k-j] * d_resp[j][0] for j in range(1, k+1)))
        repair_lp[(e,d)] = dict(mask=mask0, rd=rd0, propensity=md0,
                               y_response=y_resp, dose_response=d_resp, kernel=np.array(kernel))
        print("\n", e, d, "n", int(mask0.sum()), "residual_sd", float(np.nanstd(rd0)))
        print("lag | outcome response (cluster SE) | subsequent dose response | deconvolved effect")
        for k in range(6):
            dr = d_resp[k] if k < len(d_resp) else (np.nan,np.nan)
            print(k+1, np.round(y_resp[k],4), np.round(dr,4), round(kernel[k],4))
print("Elapsed seconds:", round(time.time()-repair_start,1))
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 16: six-hour local projections\n")
    f.write("Two-fold, stay-separated HGB adjustment using only pre-dose history. Current main reading must be observed. Estimated outcome and subsequent-dose impulse responses, then deconvolved fixed-dose response kernels; interpretation requires conditionally exogenous dose innovations.\n")
    for key, val in repair_lp.items():
        f.write(str(key)+" y_response="+str(val["y_response"])+" dose_response="+str(val["dose_response"])+" kernel="+str(val["kernel"])+"\n")
    f.write("Counts: 16 Python runs, 3 forward passes.\n")
```