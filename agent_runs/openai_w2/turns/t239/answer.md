The NE refit reduces the difficult cohort’s NLL increase from 0.00617 to 0.00558, but it still exceeds the budget there. I won’t treat that as sufficient. I’ll examine a quieter insulin decoder at the same doubled-dose sensitivity; lower held-out residuals with sensible half-dose sensitivity would justify another controlled test.

```python
fo205 = pd.read_csv("/work/fwd_25/factual_original.csv")
fe205 = pd.read_csv("/work/fwd_25/factual_edited.csv")
keys205 = ["stay_id","hour","event","observed"]
assert fo205[keys205].equals(fe205[keys205])
paired205 = fo205[keys205].copy()
paired205["delta_nll"] = fe205["nll"] - fo205["nll"]
print("FORWARD25 EXACT NLL DELTA", paired205["delta_nll"].mean())
print("BY EVENT", paired205.groupby("event")["delta_nll"].mean().to_dict())

ins_double205 = probe_labels202 == probe_keys171.index(("insulin",2.0))
ins_half205 = probe_labels202 == probe_keys171.index(("insulin",0.5))
current_probe205 = D147 @ Bcurrent202
target_ins205 = float(current_probe205[ins_double205,0].mean())
print("CURRENT INSULIN DOUBLE/HALF/CROSS-NE PROBE",
      target_ins205,
      float(current_probe205[ins_half205,0].mean()),
      float(current_probe205[ne_double202,0].mean()))
print("CURRENT INSULIN OLD/NEW FACTUAL RMS",
      rms202((X147 @ Bcurrent202)[:,0]),
      rms202((X202 @ Bcurrent202)[:,0]))

ins_fits205, ins_rows205 = {}, []
for pw205 in [0.0003,0.003,0.01,0.03,0.1,0.3]:
    for ridge205 in [1e-7,1e-6,1e-5,1e-4]:
        b205 = fit_zero_fact171(Hpool202,Hp171,Gp171,pw205,ridge205)
        s205 = float(np.mean((D147 @ b205)[ins_double205,0]))
        if s205 * target_ins205 <= 0:
            continue
        b205[:,0] *= target_ins205/s205
        w205 = fold_decoder147(b205)
        ins_fits205[(pw205,ridge205)] = (b205.copy(),w205.copy())
        p205 = D147 @ b205
        ins_rows205.append({
            "pw":pw205,"ridge":ridge205,
            "old_RMS":rms202((X147 @ b205)[:,0]),
            "new_RMS":rms202((X202 @ b205)[:,0]),
            "double":float(p205[ins_double205,0].mean()),
            "half":float(p205[ins_half205,0].mean()),
            "cross_NE":float(p205[ne_double202,0].mean())
        })
ins_table205 = pd.DataFrame(ins_rows205)
print("\nMATCHED-SENSITIVITY INSULIN FITS")
print(ins_table205.to_string(index=False,float_format=lambda x:f"{x:.6g}"))

ins_cv205 = {key:{"old":np.zeros(len(X147)),"new":np.zeros(len(X202)),
                 "probe":np.zeros(len(D147))} for key in cv_keys203}
for fold205 in range(5):
    ft205 = fact_folds172 != fold205
    nt205 = new_folds203 != fold205
    pt205 = probe_folds172 != fold205
    xf205,xn205,dp205,yp205 = X147[ft205],X202[nt205],D147[pt205],Y147[pt205]
    hf205 = (xf205.T@xf205+xn205.T@xn205)/(len(xf205)+len(xn205))
    hp205,gp205 = dp205.T@dp205/len(dp205),dp205.T@yp205/len(dp205)
    im205 = ins_double205 & pt205
    reference205 = float(current_probe205[im205,0].mean())
    for key205 in cv_keys203:
        b205 = fit_zero_fact171(hf205,hp205,gp205,*key205)
        b205[:,0] *= reference205/float((D147@b205)[im205,0].mean())
        ins_cv205[key205]["old"][~ft205] = X147[~ft205]@b205[:,0]
        ins_cv205[key205]["new"][~nt205] = X202[~nt205]@b205[:,0]
        ins_cv205[key205]["probe"][~pt205] = D147[~pt205]@b205[:,0]
ins_cv_table205 = pd.DataFrame([
    {"pw":key[0],"ridge":key[1],
     "old_CV_RMS":rms202(p["old"]),"new_CV_RMS":rms202(p["new"]),
     "double":float(p["probe"][ins_double205].mean()),
     "half":float(p["probe"][ins_half205].mean())}
    for key,p in ins_cv205.items()
])
print("\nINSULIN STAY-HELD-OUT CV")
print(ins_cv_table205.to_string(index=False,float_format=lambda x:f"{x:.6g}"))
print("WEIGHTS STILL CANDIDATE204",
      np.array_equal(np.asarray(load("core.b_in.weight")),candidate_B204))

with open("/work/notes.md","a") as f:
    f.write("\n## Forward25 / run205: NE refit helps but is not sufficient\n"
            f"Difficult-cohort factual delta {paired205['delta_nll'].mean():.8f}; "
            "still above 0.005 locally. Investigating zero-factual-target insulin "
            "fits at matched existing doubled-dose probe sensitivity, without "
            "weight changes in this run. Held-out diagnostics:\n"
            + ins_cv_table205.to_string(index=False) + "\n")
```
