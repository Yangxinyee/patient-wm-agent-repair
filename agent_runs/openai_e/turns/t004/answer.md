The raw associations change sharply after adjustment, and the norepinephrine estimate even changes sign with the controls. Those regressions are not yet reliable causal evidence. Insulin dosing jumps near specific glucose thresholds, so I’ll test those discontinuities: an outcome jump accompanying a dose jump, without a jump in observation availability, would provide stronger evidence.

```python
# Combine same-hour sensors only for identification; retain sensor flags.
for p in [panel, tpanel]:
    p["glucose_any"] = p[["glucose_finger","glucose_lab"]].mean(axis=1)
    p["map_any"] = pd.concat([p.map_arterial, p.map_cuff-3], axis=1).mean(axis=1)
    g = p.groupby("stay_id", sort=False)
    for c in ["glucose_any","map_any"]:
        p[c+"_next"] = g[c].shift(-1)
        p[c+"_prev"] = g[c].shift(1)

print("=== EXACT DOSE THRESHOLD TABLES ===")
for running,drug,lo,hi in [
    ("glucose_finger","insulin",145,155),
    ("glucose_finger","insulin",175,185),
    ("map_arterial","norepinephrine",60,75),
    ("map_cuff","norepinephrine",60,75)]:
    s = panel[(panel.t>=0)&(panel.t<24)&panel[running].between(lo,hi)].copy()
    s["treated"] = (s[drug]>0).astype(float)
    tab = s.groupby(running).agg(n=(drug,"size"), rate=(drug,"mean"),
                                  treated=("treated","mean"))
    print(running, drug, "\n"+tab.round(3).to_string())

def rd_estimate(running, outcome, drug, cutoff, window, adjust=False, order=1):
    s = panel[(panel.t>=0)&(panel.t<24)&
              panel[running].between(cutoff-window,cutoff+window-1)&
              panel[outcome+"_next"].notna()].copy()
    x = (s[running].to_numpy()-(cutoff-.5))/window
    j = (s[running].to_numpy()>=cutoff).astype(float)
    cols=[np.ones(len(s)),x,x*j]
    if order==2:
        cols += [x*x,x*x*j]
    if adjust:
        cols += [s.t.to_numpy()/24]
        for c in MEAS:
            sp=NUM[(c,"value")]
            for suffix in ["_prev","_base"]:
                v=(s[c+suffix].fillna(sp["median"]).to_numpy()-sp["median"])/sp["iqr"]
                cols += [v,v*v]
            cols += [s[c].notna().to_numpy().astype(float)]
        cols += [s[c+"_prev"].to_numpy() for c in DRUGS]
    X=np.column_stack(cols+[j])
    y=s[outcome+"_next"].to_numpy()
    d=s[drug].to_numpy()
    by,sy,_=ols_cluster(X,y,s.stay_id.to_numpy())
    bd,sd,_=ols_cluster(X,d,s.stay_id.to_numpy())
    effect=by[-1]/bd[-1]
    _,se,_=ols_cluster(X,y-effect*d,s.stay_id.to_numpy())
    # Check whether next-hour outcome availability changes at the cutoff.
    all_s=panel[(panel.t>=0)&(panel.t<24)&
                panel[running].between(cutoff-window,cutoff+window-1)]
    xa=(all_s[running].to_numpy()-(cutoff-.5))/window
    ja=(all_s[running].to_numpy()>=cutoff).astype(float)
    Xa=np.column_stack([np.ones(len(all_s)),xa,xa*ja,ja])
    bm,sm,_=ols_cluster(Xa,all_s[outcome+"_next"].notna().to_numpy().astype(float),
                        all_s.stay_id.to_numpy())
    return [len(s),bd[-1],sd[-1],by[-1],effect,se[-1]/abs(bd[-1]),bm[-1]]

print("\n=== FUZZY RD: EFFECT PER UNIT/HOUR ===")
rows=[]
for cut in [150,180,220]:
    for w in [4,8,12]:
        for adj in [False,True]:
            rows.append([cut,w,adj]+rd_estimate(
                "glucose_finger","glucose_any","insulin",cut,w,adj))
rd_table=pd.DataFrame(rows,columns=[
    "cutoff","window","history_adj","n","dose_jump","dose_SE",
    "outcome_jump","effect","effect_SE","observation_jump"])
print(rd_table.round(4).to_string(index=False))

# Check the same insulin thresholds on the independent lab sensor.
rows=[]
for cut in [150,180,220]:
    rows.append([cut]+rd_estimate("glucose_finger","glucose_lab","insulin",cut,8,True))
print("\n=== RD WITH NEXT LAB OUTCOME ===")
print(pd.DataFrame(rows,columns=["cutoff","n","dose_jump","dose_SE","outcome_jump",
                                "effect","effect_SE","observation_jump"]).round(4).to_string(index=False))

with open("/work/notes.md","a") as f:
    f.write("\n## Run 4: selection and threshold identification\n"
            "Unadjusted insulin/glucose slope +23.36, adjusted about -1.92 per unit/hour. "
            "NE/MAP slope changes from -30.02 raw to -1.90 with current reading and +2.25 with history. "
            "These are not accepted as causal. Insulin dose thresholds near 150/180/220 motivate fuzzy RD. "
            "Test independent lab outcomes and observation availability to diagnose selection artifacts.\n"
            "Python runs: 4. Forward requests: 0.\n")
```
