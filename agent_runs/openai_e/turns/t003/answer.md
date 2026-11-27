Infusions start five minutes after the hourly readings, which gives a clear treatment-before-outcome ordering. Doses are discretised, but I have not yet established whether their variation is informative. I’ll build an hourly panel and compare unadjusted associations with history-adjusted estimates; stable estimates across controls would support, but not prove, a causal interpretation.

```python
from scipy import stats
from sklearn.linear_model import Ridge

MEAS = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
DRUGS = ["insulin", "norepinephrine"]

def make_panel(data):
    m = data[data.event.isin(MEAS)].copy()
    m["t"] = np.rint(m.hour).astype(int)
    p = m.pivot_table(index=["stay_id", "t"], columns="event", values="value")
    ids = data.stay_id.unique()
    index = pd.MultiIndex.from_product([ids, range(-1, 25)], names=["stay_id", "t"])
    p = p.reindex(index)
    d = data[data.event.isin(DRUGS)].copy()
    d["t"] = np.floor(d.hour).astype(int)
    rates = d.pivot_table(index=["stay_id", "t"], columns="event", values="rate", aggfunc="sum")
    p = p.join(rates).reset_index()
    p[DRUGS] = p[DRUGS].fillna(0)
    g = p.groupby("stay_id", sort=False)
    for c in MEAS:
        p[c + "_last"] = g[c].ffill()
        observed_t = p["t"].where(p[c].notna())
        p[c + "_age"] = p["t"] - observed_t.groupby(p.stay_id, sort=False).ffill()
        p[c + "_prev"] = g[c + "_last"].shift(1)
        p[c + "_next"] = g[c].shift(-1)
        p[c + "_base"] = p[c + "_last"].where(p.t <= 0).groupby(p.stay_id, sort=False).transform("first")
    for c in DRUGS:
        p[c + "_prev"] = g[c].shift(1).fillna(0)
    return p

panel = make_panel(df)
tpanel = make_panel(tune)
rng = np.random.default_rng(731)
validation_ids = rng.permutation(tune.stay_id.unique())[:100].tolist()

print("=== TIMING AND RATES ===")
print("Distinct hours:", sorted(df.hour.unique())[:65])
for drug in DRUGS:
    a = df.loc[df.event == drug, "rate"]
    print(drug, "quantiles:", a.quantile([0,.01,.1,.5,.9,.99,1]).round(4).to_dict())
    print("Most common rates:", a.value_counts().head(12).to_dict())
print("Panel shape:", panel.shape)
print("Paired reading differences:")
for a,b in [("glucose_finger","glucose_lab"),("map_arterial","map_cuff")]:
    r = panel[[a,b]].dropna()
    print(a, "-", b, "n", len(r), "mean", round((r[a]-r[b]).mean(),3),
          "sd", round((r[a]-r[b]).std(),3), "corr", round(r[a].corr(r[b]),4))

def ols_cluster(X, y, ids):
    beta = np.linalg.lstsq(X, y, rcond=1e-10)[0]
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X, rcond=1e-12)
    _, gi = np.unique(ids, return_inverse=True)
    sums = np.zeros((gi.max()+1, X.shape[1]))
    np.add.at(sums, gi, X * resid[:,None])
    cov = bread @ (sums.T @ sums) @ bread
    return beta, np.sqrt(np.maximum(np.diag(cov),0)), np.std(resid)

print("\n=== ONE-HOUR RATE COEFFICIENTS (clustered SE) ===")
for outcome, drug in [("glucose_finger","insulin"),("map_arterial","norepinephrine")]:
    s = panel[(panel.t >= 0) & (panel.t < 24) &
              panel[outcome].notna() & panel[outcome+"_next"].notna()].copy()
    y = s[outcome+"_next"].to_numpy()
    rate = s[drug].to_numpy()
    scale = NUM[(outcome,"value")]["iqr"]
    v = (s[outcome].to_numpy() - NUM[(outcome,"value")]["median"])/scale
    base = [np.ones(len(s))]
    designs = {"unadjusted": base,
               "current+time": base+[v,v*v,v*v*v,s.t.to_numpy()/24]}
    history = designs["current+time"].copy()
    for c in MEAS:
        sp = NUM[(c,"value")]
        for suffix in ["_last","_prev","_base"]:
            z = (s[c+suffix].fillna(sp["median"]).to_numpy()-sp["median"])/sp["iqr"]
            history.extend([z,z*z])
        history.extend([s[c].notna().to_numpy().astype(float),
                        s[c+"_age"].fillna(26).to_numpy()/24])
    history += [s[c+"_prev"].to_numpy() for c in DRUGS]
    history += [s[c].to_numpy() for c in DRUGS if c != drug]
    designs["history+other_readings"] = history
    for label, cols in designs.items():
        X = np.column_stack(cols+[rate])
        b,se,sd = ols_cluster(X,y,s.stay_id.to_numpy())
        print(outcome, drug, label, "n",len(s), "beta",round(b[-1],4),
              "SE",round(se[-1],4),"resid_sd",round(sd,3))

print("\n=== DOSE BY CURRENT READING ===")
for outcome, drug, width in [("glucose_finger","insulin",10),("map_arterial","norepinephrine",5)]:
    s = panel[(panel.t>=0)&(panel.t<24)&panel[outcome].notna()].copy()
    s["reading_group"] = (np.floor(s[outcome]/width)*width).astype(int)
    tab = s.groupby("reading_group")[drug].agg(["count","mean","std"])
    print(outcome, "\n", tab.loc[tab["count"]>1000].round(3).to_string())

with open("/work/notes.md","a") as f:
    f.write("\n## Run 3: hourly panel\n"
            "Readings occur on integer hours; infusions start at hour + 0.0833 and last one hour. "
            "Built full panels with absent infusion=0, prior reading registers, ages, and next-hour outcomes. "
            "Initial regressions are diagnostics, not yet accepted causal estimates.\n"
            "Validation IDs chosen randomly once (seed 731), 100 tuning stays.\n"
            "Python runs: 3. Forward requests: 0.\n")
```
