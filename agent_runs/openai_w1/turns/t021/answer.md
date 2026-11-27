The run stopped on an indexing error before producing the six-hour results; it tells us nothing new about drug effects. I’ll correct the exposure-hour indices and rerun the same analysis. The hypothesis and validation criteria are unchanged.

```python
repair_j = repair_j - 1  # indices 4..19 = ICU hours 3..18, leaving six future hours

def repair_build_features(si, ji):
    cols, names = [ji.astype(float)-1], ["hour"]
    for e in repair_events:
        for lag in range(4):
            cols.append(repair_last[e][si, ji-lag])
            names.append(e+"_last_lag"+str(lag))
        cols += [repair_age[e][si,ji], np.isfinite(P[e][si,ji]).astype(float),
                 repair_means[e][si,ji], P[e][si,0], P[e][si,1]]
        names += [e+s for s in ["_age","_present","_mean","_initial","_hour0"]]
    for d in repair_drugs:
        for lag in range(1,5):
            cols.append(repair_rates[d][si,ji-lag])
            names.append(d+"_lag"+str(lag))
        cols.append(np.mean(repair_rates[d][:,:5],axis=1)[si])
        names.append(d+"_earlymean")
    return np.column_stack(cols).astype(np.float32), names

repair_C, repair_feature_names = repair_build_features(repair_s, repair_j)
repair_lp = {}
repair_start = time.time()
with threadpool_limits(limits=8):
    for e, d in [("glucose_finger","insulin"), ("map_arterial","norepinephrine")]:
        mask0 = np.isfinite(P[e][repair_s,repair_j])
        target_d = repair_rates[d][repair_s,repair_j]
        md0 = repair_crossfit(target_d,mask0,repair_C,repair_fold)
        rd0 = target_d-md0
        y_resp, d_resp = [], [(1.0,0.0)]
        for k in range(1,7):
            yy = P[e][repair_s,repair_j+k]
            mask = mask0 & np.isfinite(yy)
            my = repair_crossfit(yy,mask,repair_C,repair_fold)
            y_resp.append(repair_cluster_beta(rd0[mask],(yy-my)[mask],repair_s[mask]))
            if k <= 5:
                dd = repair_rates[d][repair_s,repair_j+k]
                mf = repair_crossfit(dd,mask0,repair_C,repair_fold)
                d_resp.append(repair_cluster_beta(rd0[mask0],(dd-mf)[mask0],repair_s[mask0]))
        kernel = []
        for k in range(6):
            kernel.append(y_resp[k][0]-sum(kernel[k-j]*d_resp[j][0] for j in range(1,k+1)))
        repair_lp[(e,d)] = dict(mask=mask0,rd=rd0,propensity=md0,
                               y_response=y_resp,dose_response=d_resp,kernel=np.array(kernel))
        print("\n",e,d,"n",int(mask0.sum()),"residual_sd",round(float(np.nanstd(rd0)),5))
        for k in range(6):
            dr = d_resp[k] if k<len(d_resp) else (np.nan,np.nan)
            print("lag",k+1,"outcome",np.round(y_resp[k],4),
                  "dose",np.round(dr,4),"fixed-dose kernel",round(kernel[k],4))
print("Elapsed seconds:",round(time.time()-repair_start,1))
with open("/work/notes.md","a") as f:
    f.write("\n## Runs 16–17: six-hour local projections\nRun 16 failed because exposure indices extended one hour too far; corrected in run 17. No edits.\n")
    f.write("Two-fold stay-separated HGB adjustment using pre-dose history, current main reading observed. Deconvolution separates later policy-dose changes from an estimated fixed-dose response. Requires conditional exogeneity; not established merely by this regression.\n")
    for key,val in repair_lp.items():
        f.write(str(key)+" y_response="+str(val["y_response"])+" dose_response="+str(val["dose_response"])+" kernel="+str(val["kernel"])+"\n")
    f.write("Counts: 17 Python runs, 3 forward passes.\n")
```