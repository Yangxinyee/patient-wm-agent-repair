Both glucose sensors show an insulin effect near −1.9 per rate unit, and both MAP sensors show a positive norepinephrine effect. The dose-range printout was incorrect: the regression includes an intercept, which I mistakenly indexed as a drug coefficient. The dynamic calculation also stopped on a format error. I’ll correct both before drawing further conclusions; stable corrected estimates would support the current effect hypothesis.

```python
print("=== CORRECTED SENSOR EFFECTS ===")
for event,(bb,ss,nn) in sensor_checks.items():
    print(event, "insulin",round(float(bb[1]),4),"SE",round(float(ss[1]),4),
          "NE",round(float(bb[2]),4),"SE",round(float(ss[2]),4),"n",nn)

print("\n=== CORRECTED DOSE-RANGE CHECKS ===")
for outcome,drug,coefidx in [("G_next","insulin",1),("M_next","norepinephrine",2)]:
    rr = residuals[outcome]
    rawrate = panel[drug].to_numpy()[testmask]
    valid = np.isfinite(rr) & np.isfinite(rt).all(axis=1)
    for lo,hi in ([(0,2),(2,5),(5,50)] if drug=="insulin" else [(0,.15),(.15,.4),(.4,2)]):
        vm = valid & (rawrate>=lo) & (rawrate<hi)
        if vm.sum()>500:
            bb,ss,nn = clustered_ols(rt[vm],rr[vm],testsid[vm])
            print(outcome,lo,hi,"effect",round(float(bb[coefidx]),4),
                  "SE",round(float(ss[coefidx]),4),"n",nn)

print("\n=== LOCAL-PROJECTION FORMAT ===")
for key in [("G",1),("M",1),("I",0),("N",0),("G",2)]:
    print(key, repr(lp[key])[:450])

print("\n=== MODEL-PROBE DETAILS ===")
print("first 15 state rows:")
print(pd.DataFrame({k:sA[k][:15] for k in
                    ["stay_id","hour","has_insulin","has_norepinephrine","n_atoms"]}).to_string(index=False))
print("first stay events:")
print(tuning.loc[tuning.stay_id.eq(eval_ids_A[0]),
                 ["hour","event","value","rate","dose","end_hour"]].head(17).to_string(index=False))
print("activation-related bias quantiles:",
      np.round(np.quantile(np.asarray(W0["attr_heads.value_bin.0.bias"]),[0,.25,.5,.75,1]),3))
print("gap shapes:",[(n,SHAPES[n]) for n in NAMES if n.startswith("gap.")])

train_probe_ids = [int(i) for i in all_train_ids[:100]]
print("\ntrain probe IDs:",train_probe_ids)
with open("/work/notes.md","a") as f:
    f.write("\n## Sensor checks and correction (runs 8–9)\n"
            "- Sensor-specific held-out next-hour estimates: glucose_finger insulin -1.8383 (SE .1026), "
            "glucose_lab -1.9344 (.1744); map_arterial NE +3.4086 (.2191), map_cuff +2.8671 (.4111).\n"
            "- Cross-drug coefficients are small relative to their uncertainty.\n"
            "- Corrected a diagnostic indexing error: clustered_ols includes an intercept. "
            "The initial dose-stratum printouts were intercepts, not drug effects, and must be discarded.\n"
            "- Run 8 dynamic-alternative calculation stopped before completion; no edits were made.\n")
```
