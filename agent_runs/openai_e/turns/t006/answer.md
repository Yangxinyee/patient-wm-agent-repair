The original model predicts insulin shifts of −1.847 and +0.777 mg/dL, and norepinephrine shifts of −0.498 and −0.600 mmHg. These are model outputs, not truth. I’ll estimate how insulin effects persist and investigate norepinephrine with longer histories; agreement across those analyses would justify targets for the edits.

```python
torch.set_num_threads(8)
VALID_IDS = list(range(2020000000, 2020000100))
baseline_rollout = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("=== ROLLOUT FILE ===")
print(baseline_rollout.head(3).to_string(index=False))

# Add strictly pre-dose history, preserving observation/missingness information.
for p in [panel,tpanel]:
    g=p.groupby("stay_id",sort=False)
    for c in MEAS+DRUGS:
        source=c+"_last" if c in MEAS else c
        for lag in range(1,5):
            p[c+f"_lag{lag}"]=g[source].shift(lag)
    for horizon in range(1,9):
        p["glucose_any_f"+str(horizon)]=g.glucose_any.shift(-horizon)
        p["insulin_f"+str(horizon)]=g.insulin.shift(-horizon)

print("\n=== NE ESTIMATES WITH LONGER PRE-DOSE HISTORY ===")
s=panel[(panel.t>=4)&(panel.t<24)&panel.map_arterial.notna()&
        panel.map_arterial_next.notna()].copy()
y=s.map_arterial_next.to_numpy()
cols=[np.ones(len(s)),s.t.to_numpy()/24]
for c in MEAS:
    sp=NUM[(c,"value")]
    v=(s[c+"_last"].fillna(sp["median"]).to_numpy()-sp["median"])/sp["iqr"]
    cols += [v,v*v,v*v*v,s[c].notna().to_numpy().astype(float),
             s[c+"_age"].fillna(26).to_numpy()/24]
for nlags in [0,1,2,4]:
    cc=cols.copy()
    for lag in range(1,nlags+1):
        for c in MEAS:
            sp=NUM[(c,"value")]
            v=(s[c+f"_lag{lag}"].fillna(sp["median"]).to_numpy()-sp["median"])/sp["iqr"]
            cc += [v,v*v]
        for c in DRUGS:
            v=s[c+f"_lag{lag}"].fillna(0).to_numpy()
            cc += [v,v*v]
    X=np.column_stack(cc+[s.insulin.to_numpy(),s.norepinephrine.to_numpy()])
    b,se,sd=ols_cluster(X,y,s.stay_id.to_numpy())
    print("lags",nlags,"n",len(s),"NE beta",round(b[-1],4),
          "SE",round(se[-1],4),"residual_sd",round(sd,3))

print("\n=== EXAMPLE HIGH-NE TRAJECTORY ===")
example_id=panel.loc[(panel.norepinephrine>.8)&(panel.t>12),"stay_id"].iloc[0]
print(panel.loc[panel.stay_id==example_id,
                ["t","map_arterial","map_cuff","norepinephrine"]].round(3).to_string(index=False))

# Pooled local discontinuities at 180 and 220; omit the rounded boundary value.
def pooled_rd(horizon, window=12, order=1):
    pieces=[]
    for cut in [180,220]:
        ss=panel[(panel.t>=0)&(panel.t+horizon<=24)&
                 panel.glucose_finger.between(cut-window,cut+window)&
                 (panel.glucose_finger!=cut)&
                 panel["glucose_any_f"+str(horizon)].notna()].copy()
        ss["cut"]=cut
        pieces.append(ss)
    ss=pd.concat(pieces,ignore_index=True)
    x=(ss.glucose_finger.to_numpy()-ss.cut.to_numpy())/window
    j=(x>0).astype(float)
    c=(ss.cut.to_numpy()==220).astype(float)
    cols=[np.ones(len(ss)),c,x,x*c,x*j,x*j*c,ss.t.to_numpy()/24]
    if order==2:
        cols += [x*x,x*x*c,x*x*j,x*x*j*c]
    for m in MEAS:
        sp=NUM[(m,"value")]
        v=(ss[m+"_prev"].fillna(sp["median"]).to_numpy()-sp["median"])/sp["iqr"]
        cols += [v,v*v,ss[m].notna().to_numpy().astype(float)]
    for d in DRUGS:
        cols += [ss[d+"_prev"].to_numpy()]
    X=np.column_stack(cols+[j])
    ids=ss.stay_id.to_numpy()
    by,sy,_=ols_cluster(X,ss["glucose_any_f"+str(horizon)].to_numpy(),ids)
    jumps=[]
    for lag in range(horizon):
        v=ss.insulin.to_numpy() if lag==0 else ss["insulin_f"+str(lag)].to_numpy()
        bd,_,_=ols_cluster(X,v,ids)
        jumps.append(bd[-1])
    return len(ss),by[-1],sy[-1],jumps

dynamic_rd={}
print("\n=== INSULIN LOCAL-PROJECTION DISCONTINUITIES ===")
for h in range(1,7):
    r=pooled_rd(h)
    dynamic_rd[h]=r
    print("h",h,"n",r[0],"glucose_jump",round(r[1],4),"SE",round(r[2],4),
          "dose_jumps",[round(v,4) for v in r[3]])

from scipy.optimize import least_squares
def rd_dynamic_residual(par):
    beta,a=par
    return np.array([(dynamic_rd[h][1] -
                      beta*sum(a**(h-1-j)*v for j,v in enumerate(dynamic_rd[h][3])))/
                     dynamic_rd[h][2] for h in dynamic_rd])
dyn_fit=least_squares(rd_dynamic_residual,[-2,.8],bounds=([-10,0],[5,1.2]))
print("RD dynamic fit [beta, persistence]:",dyn_fit.x.round(5),
      "standardised residuals:",rd_dynamic_residual(dyn_fit.x).round(3))

with open("/work/notes.md","a") as f:
    f.write("\n## Forward 1: original rollout, first 100 tuning stays\n"
            "Insulin x2 -1.847; insulin half +0.777; NE x2 -0.498; NE half -0.600. "
            "Insulin reaches 554 stay-hours/60 stays; NE 281/26. "
            "These are predictions, not truth.\n"
            "RD roughly -2 per additional insulin unit/hour, with serious observation jump at 150; "
            "prefer thresholds 180/220 and check temporal persistence. "
            "Boundary observations are mixtures because measurements are rounded.\n"
            "Python runs: 5. Forward requests: 1.\n")
```
