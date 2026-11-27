The original factual NLL is 1.59524 on these 100 tuning stays. That is the baseline to protect. Next I’ll identify which glucose reading drives insulin decisions and test for outcome changes at dose-policy thresholds. A dose jump without a corresponding outcome jump would argue against a large insulin effect.

```python
PY_RUNS += 1
FWD_RUNS = 2
from scipy import linalg

def ffill_stay(s):
    return s.groupby(level=0).ffill()

g_candidates = {
    "finger_first": ffill_stay(pan.glucose_finger.combine_first(pan.glucose_lab)),
    "lab_first": ffill_stay(pan.glucose_lab.combine_first(pan.glucose_finger)),
    "minimum": ffill_stay(pan[["glucose_finger","glucose_lab"]].min(axis=1)),
    "maximum": ffill_stay(pan[["glucose_finger","glucose_lab"]].max(axis=1)),
}
policy_errors = {}
print("=== INSULIN START POLICY: ERROR OF score > 150 ===")
for name, s in g_candidates.items():
    ok = valid_hours & s.notna().to_numpy()
    err = ((s.to_numpy()[ok] > 150) != (pan.insulin.to_numpy()[ok] > 0)).mean()
    policy_errors[name] = err
    print(name, round(err,6))
policy_name = min(policy_errors, key=policy_errors.get)
policy_g = g_candidates[policy_name]
fresh_g = pan[["glucose_finger","glucose_lab"]].notna().any(axis=1).to_numpy()
print("Best candidate:", policy_name)

print("\n=== LOCAL INSULIN POLICY AND NEXT GLUCOSE ===")
delta_g = next_obs.glucose_finger - last.glucose_finger
tab = pd.DataFrame({"score":policy_g, "rate":pan.insulin,
                    "next_glucose":next_obs.glucose_finger,
                    "change":delta_g, "next_observed":next_obs.glucose_finger.notna()})
tab = tab.loc[valid_hours & fresh_g]
tab = tab.groupby("score").agg(n=("rate","size"), rate=("rate","mean"),
                              next_glucose=("next_glucose","mean"),
                              change=("change","mean"),
                              next_observed=("next_observed","mean"))
wanted = list(range(145,156)) + list(range(176,186)) + list(range(216,226))
print(tab.reindex(wanted).round(3).to_string())

stay_codes = pd.factorize(pan.index.get_level_values("stay_id"))[0]
previous_g = last.glucose_finger.groupby(level=0).shift(1).to_numpy()
baseline_g = pan.glucose_finger.groupby(level=0).transform("first").to_numpy()

def local_rd(score, dose, outcome, cutoff, bandwidth, eligible):
    score = np.asarray(score, dtype=float)
    dose = np.asarray(dose, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    r = score - cutoff
    ok = eligible & np.isfinite(r) & np.isfinite(outcome) & (np.abs(r)<bandwidth)
    rr = r[ok]
    zz = (rr>0).astype(float)
    w = 1 - np.abs(rr)/bandwidth
    X = np.column_stack([np.ones(len(rr)), rr, zz, rr*zz])
    gram = X.T @ (w[:,None]*X)
    inv = np.linalg.pinv(gram)
    bD = inv @ (X.T @ (w*dose[ok]))
    bY = inv @ (X.T @ (w*outcome[ok]))
    jump_d, jump_y = bD[2], bY[2]
    beta = jump_y/jump_d if abs(jump_d)>1e-5 else np.nan
    res = (outcome[ok]-X@bY) - beta*(dose[ok]-X@bD)
    infl = (X@inv[:,2])*w*res/jump_d
    sums = np.bincount(stay_codes[ok], weights=infl, minlength=len(train_ids))
    se = np.sqrt(np.sum(sums*sums))
    return {"cutoff":cutoff,"bw":bandwidth,"n":int(ok.sum()),
            "dose_jump":jump_d,"outcome_jump":jump_y,
            "effect_per_rate":beta,"cluster_se":se}

rd_results = []
for threshold in [150.5,180.5,220.5,260.5]:
    for bw in [5,10,15,20]:
        rd_results.append(local_rd(
            policy_g, pan.insulin, next_obs.glucose_finger,
            threshold,bw,valid_hours & fresh_g))
rd_results = pd.DataFrame(rd_results)
print("\n=== LOCAL-LINEAR THRESHOLD ESTIMATES (GLUCOSE PER INSULIN UNIT/HOUR) ===")
print(rd_results.round(4).to_string(index=False))

placebos = []
for label, y in [("previous_glucose",previous_g),("baseline_glucose",baseline_g),
                 ("next_reading_probability",next_obs.glucose_finger.notna().astype(float))]:
    row = local_rd(policy_g,pan.insulin,y,150.5,10,valid_hours & fresh_g)
    row["outcome"] = label
    placebos.append(row)
print("\n=== THRESHOLD PLACEBO / OBSERVATION CHECKS ===")
print(pd.DataFrame(placebos)[["outcome","n","dose_jump","outcome_jump","cluster_se"]]
      .round(4).to_string(index=False))

m_candidates = {
    "arterial_first": ffill_stay(pan.map_arterial.combine_first(pan.map_cuff)),
    "cuff_first": ffill_stay(pan.map_cuff.combine_first(pan.map_arterial)),
    "minimum": ffill_stay(pan[["map_arterial","map_cuff"]].min(axis=1)),
}
print("\n=== MAP POLICY SUMMARY ===")
for name,s in m_candidates.items():
    d = pd.DataFrame({"score":s,"dose":pan.norepinephrine}).loc[valid_hours]
    d["band"] = pd.cut(d.score, [20,40,50,60,65,70,75,80,90,110])
    print(name, d.groupby("band",observed=True).dose.mean().round(3).to_dict())

baseline_fact = pd.read_csv("/work/fwd_2/factual_original.csv")
print("\nFactual CSV columns:",baseline_fact.columns.tolist())
print(baseline_fact.head(2).to_string(index=False))

note(f"""## Run {PY_RUNS}; forward requests used: {FWD_RUNS}
Original factual NLL on first 100 tuning stays: 1.59524 (4,903 readings).
Insulin threshold policy best explained by {policy_name};
classification error {policy_errors[policy_name]:.6f}.
Computed local-linear RD estimates at 150.5, 180.5, 220.5, 260.5,
with stay-clustered uncertainty and pre-outcome/observation checks.
RD table:\n{rd_results.round(4).to_string(index=False)}
""")
```
