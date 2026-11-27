The record joins returned no matching doses or readings, so those zero-dose fits are invalid—not evidence of absent drug effects. I’ll fix the split and timestamp matching, then repeat the comparison and rate fit. Nonzero matched doses must be recovered before any feature calibration is usable.

```python
print("=== SPLIT AND CLOCK CHECK ===")
print("state IDs", np.unique(rr33_stays)[:5], "dtype", np.asarray(rr33_stays).dtype)
print("train IDs", df.stay_id.unique()[:3], "tuning IDs", tune.stay_id.unique()[:3])
print("state hours", np.asarray(rr33_hours)[r234_indices],
      "head next hours", r237_lh["hours49"] + r237_lh["dt_np49"]/3600)
print("drug-bundle hours", np.asarray(rr33_hours)[np.asarray(indices151)[:8]])

r238_ids = np.unique(rr33_stays)
r238_records = pd.concat([df[df.stay_id.isin(r238_ids)],
                          tune[tune.stay_id.isin(r238_ids)]], ignore_index=True)
r238_records["hour_key"] = r238_records.hour.round(4)
r238_rt = r238_records[r238_records.event.isin(["insulin","norepinephrine"])].pivot_table(
    index=["stay_id","hour_key"], columns="event", values="rate", aggfunc="sum")
r238_ri = pd.MultiIndex.from_arrays(
    [np.asarray(rr33_stays)[np.asarray(indices151)],
     np.round(np.asarray(rr33_hours)[np.asarray(indices151)],4)],
    names=["stay_id","hour_key"])
r238_doses = r238_rt.reindex(r238_ri).reindex(
    columns=["insulin","norepinephrine"]).fillna(0).to_numpy()
print("matched record rows", len(r238_records),
      "dose means", r238_doses.mean(axis=0), "nonzero counts", (r238_doses>0).sum(axis=0))
print("selected-dose RMS error",
      np.sqrt(np.mean((r238_doses[np.arange(320), np.asarray(drug151,int)] -
                       np.asarray(dc151))**2)))

r238_ref = pd.concat([pd.read_csv(p) for p in Path("/work").glob("fwd_*/factual_original.csv")],
                    ignore_index=True).drop_duplicates(["stay_id","hour","event"])
r238_rows = []
for r238_j,r238_event in enumerate(r49_sensors):
    r238_local = r235_result[r238_event]["M"].mean(axis=1)
    r238_next = r237_lh["hours49"] + r237_lh["dt_np49"]/3600
    for r238_k,r238_i in enumerate(r234_indices):
        r238_match = r238_ref[(r238_ref.stay_id==rr33_stays[r238_i]) &
                              (abs(r238_ref.hour-r238_next[r238_k])<1e-4) &
                              (r238_ref.event==r238_event)]
        if len(r238_match):
            r238_rows.append((float(rr33_hours[r238_i]),float(r238_next[r238_k]),
                              r238_event,float(r238_local[r238_k]),
                              float(r238_match.predicted_mean.iloc[0])))
r238_comparison = pd.DataFrame(r238_rows,columns=["state_hour","reading_hour","event","local","official"])
print("\n=== MATCHED HEAD CHECK ===")
print(r238_comparison.to_string(index=False))

print("\n=== CORRECTED RATE-FEATURE FITS ===")
r238_features = {}
if np.any(r238_doses):
    r238_A = trials230[(0.0,"factual")]["E"].numpy().reshape(-1,192).astype(np.float64)
    for r238_j,r238_drug in enumerate(("insulin","norepinephrine")):
        r238_D = (trials230[(0.0,r238_drug)]["E"]-
                  trials230[(0.0,"factual")]["E"]).numpy().reshape(-1,192).astype(np.float64)
        r238_y = np.repeat(r238_doses[:,r238_j],4)
        for r238_lam in (1e-7,1e-6,1e-5):
            r238_v = np.linalg.solve(
                (r238_A.T@r238_A+r238_D.T@r238_D)/len(r238_A)+r238_lam*np.eye(192),
                r238_D.T@r238_y/len(r238_A))
            r238_features[(r238_drug,r238_lam)] = r238_v
            r238_pred = r238_D@r238_v
            print(r238_drug,r238_lam,
                  "factual RMS",float(np.sqrt(np.mean((r238_A@r238_v)**2))),
                  "contrast slope",float(r238_pred@r238_y/(r238_y@r238_y)),
                  "contrast RMSE",float(np.sqrt(np.mean((r238_pred-r238_y)**2))))

r238_physbase = {k: torch.as_tensor(v).clone() for k,v in candidate158.items()}
r238_physbase["z_in.weight"][:,30:32] = 0
r238_postresult = posterior_token154(r238_physbase, context151["u"])
print("\nposterior-token return",
      [(type(v).__name__,tuple(v.shape)) for v in r238_postresult]
      if isinstance(r238_postresult,tuple) else tuple(r238_postresult.shape))
with open("/work/notes.md","a") as r238_f:
    r238_f.write("\n## Runs 237-238: repair diagnostic joins\n"
                 "- Run 237's zero doses and empty head comparison were failed joins, not scientific results.\n"
                 "- Matching records across available splits with rounded bundle timestamps.\n"
                 "- No weights changed. Python runs: 238; forward passes: 17.\n")
```
