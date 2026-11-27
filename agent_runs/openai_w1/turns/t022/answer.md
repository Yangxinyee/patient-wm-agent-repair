The adjusted responses persist for several hours and are much larger than the original model’s reported rollouts. That makes weak learned dose effects a plausible problem. I also found a first-hour control that included the current dose; I’ll remove that leakage, check sensitivity to richer adjustment, and select representative stays for weight-edit calibration.

```python
repair_C = repair_C.copy()
for d in repair_drugs:
    repair_C[:, repair_feature_names.index(d+"_earlymean")] = \
        np.mean(repair_rates[d][:,1:4],axis=1)[repair_s]

repair_robust = {}
repair_old_params = repair_hgb_params.copy()
repair_hgb_params = dict(max_iter=180,max_leaf_nodes=31,learning_rate=.08,
                         min_samples_leaf=80,l2_regularization=10,
                         max_bins=127,early_stopping=False,random_state=119)
with threadpool_limits(limits=8):
    for e,d,alt in [("glucose_finger","insulin","glucose_lab"),
                    ("map_arterial","norepinephrine","map_cuff")]:
        mask0 = np.isfinite(P[e][repair_s,repair_j])
        dd = repair_rates[d][repair_s,repair_j]
        yy = P[e][repair_s,repair_j+1]
        mask = mask0 & np.isfinite(yy)
        md = repair_crossfit(dd,mask0,repair_C,repair_fold)
        my = repair_crossfit(yy,mask,repair_C,repair_fold)
        rd, ry = dd-md, yy-my
        beta,se = repair_cluster_beta(rd[mask],ry[mask],repair_s[mask])
        print("\nRicher adjustment:",e,d,"beta",round(beta,4),"SE",round(se,4),
              "dose residual SD",round(float(np.nanstd(rd)),4))
        pos = mask & (dd>0)
        qs = np.unique(np.quantile(dd[pos],[0,.33,.67,1]))
        for lo,hi in zip(qs[:-1],qs[1:]):
            mm = pos & (dd>=lo) & (dd<=hi if hi==qs[-1] else dd<hi)
            if mm.sum()>100:
                print(" dose",round(lo,3),"to",round(hi,3),"n",int(mm.sum()),
                      "beta/SE",np.round(repair_cluster_beta(rd[mm],ry[mm],repair_s[mm]),4))
        # Held-out simultaneous assay: omit its current value from the history.
        reduced = repair_C.copy()
        for suffix, vals in [
            ("_last_lag0",repair_last[alt][repair_s,repair_j-1]),
            ("_age",repair_age[alt][repair_s,repair_j-1]+1),
            ("_mean",repair_means[alt][repair_s,repair_j-1])]:
            reduced[:,repair_feature_names.index(alt+suffix)] = vals
        md_reduced = repair_crossfit(dd,mask0,reduced,repair_fold)
        placebo = P[alt][repair_s,repair_j]
        pmask = mask0 & np.isfinite(placebo)
        mp = repair_crossfit(placebo,pmask,reduced,repair_fold)
        pb = repair_cluster_beta((dd-md_reduced)[pmask],(placebo-mp)[pmask],repair_s[pmask])
        print(" withheld pre-dose",alt,"association beta/SE",np.round(pb,4),"n",int(pmask.sum()))
        for alt_e in [alt]:
            ya = P[alt_e][repair_s,repair_j+1]
            amask = mask0 & np.isfinite(ya)
            ma = repair_crossfit(ya,amask,repair_C,repair_fold)
            print(" next-hour",alt_e,"effect beta/SE",
                  np.round(repair_cluster_beta(rd[amask],(ya-ma)[amask],repair_s[amask]),4),
                  "n",int(amask.sum()))
        repair_robust[(e,d)] = dict(rd=rd,ry=ry,mask=mask,beta=beta,se=se,
                                   placebo=pb,propensity=md)
repair_hgb_params = repair_old_params

repair_train_counts = train.groupby(["stay_id","event"]).size().unstack(fill_value=0)
repair_rng = np.random.default_rng(9021)
repair_norep_pool = repair_train_counts.index[
    (repair_train_counts.norepinephrine>=10) & (repair_train_counts.map_arterial>=15)].to_numpy()
repair_ins_pool = repair_train_counts.index[
    (repair_train_counts.insulin>=10) & (repair_train_counts.glucose_finger>=15)].to_numpy()
repair_map_sample = repair_rng.choice(repair_norep_pool,50,replace=False)
repair_ins_sample = repair_rng.choice(np.setdiff1d(repair_ins_pool,repair_map_sample),50,replace=False)
repair_state_ids_1 = sorted(map(int,np.r_[repair_map_sample,repair_ins_sample]))
print("\nREPRESENTATIVE TRAIN STATE IDS:",json.dumps(repair_state_ids_1))
repair_npz = np.load("/work/fwd_2/states_original.npz")
print("State NPZ keys/shapes:",[(k,repair_npz[k].shape) for k in repair_npz.files])
print("Last-numeric column norms:",np.round(np.linalg.norm(load_original("last_numeric.weight"),axis=0),4))
with open("/work/notes.md","a") as f:
    f.write("\n## Run 18: causal robustness\nRemoved early-mean current-dose leakage at exposure hour 3; richer HGB adjustment and held-out simultaneous pre-dose assay diagnostics. A held-out assay association is a confounding diagnostic, not definitive proof, since clinicians may also use that assay.\n")
    for key,val in repair_robust.items():
        f.write(str(key)+" beta="+str(val["beta"])+" se="+str(val["se"])+" withheld-assay="+str(val["placebo"])+"\n")
    f.write("Selected 100 representative training stays enriched for both measured drug pathways. No edits.\nCounts: 18 Python runs, 3 forward passes.\n")
```