The original model predicts very small responses to all four dose edits. That could be correct or could reflect missed drug effects. I’ll now examine the training records’ treatment policies and subsequent changes. An effect estimate would be more credible if it survives adjustment for pre-dose readings and history, rather than appearing only in raw dose–outcome correlations.

```python
PY_RUNS += 1
FWD_RUNS = 1
measure_events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drug_events = ["insulin", "norepinephrine"]

def make_panel(df):
    d = df[df.event.isin(measure_events + drug_events)].copy()
    d["t"] = np.floor(d.hour.to_numpy() + 1e-5).astype(np.int16)
    d["number"] = np.where(d.event.isin(drug_events), d.rate, d.value)
    duplicates = d.duplicated(["stay_id", "t", "event"]).sum()
    print("Duplicate stay/hour/event rows:", duplicates)
    p = d.pivot_table(index=["stay_id", "t"], columns="event", values="number",
                      aggfunc="first", observed=True)
    ids = df.stay_id.drop_duplicates().to_numpy()
    p = p.reindex(pd.MultiIndex.from_product([ids, range(-1,25)],
                                            names=["stay_id","t"]))
    p.columns.name = None
    p[drug_events] = p[drug_events].fillna(0.0)
    return p.sort_index()

pan = make_panel(train)
tpan = make_panel(tuning)
last = pan[measure_events].groupby(level=0).ffill()
tlast = tpan[measure_events].groupby(level=0).ffill()
next_obs = pan[measure_events].groupby(level=0).shift(-1)
hours = pan.index.get_level_values("t").to_numpy()
valid_hours = (hours >= 0) & (hours <= 23)

print("=== INFUSION DISTRIBUTIONS ===")
for drug in drug_events:
    r = pan.loc[valid_hours, drug]
    positive = r[r > 0]
    d = train[train.event == drug]
    print(drug, "fraction of hours treated:", round((r>0).mean(),4))
    print("positive rate quantiles:", positive.quantile([0,.01,.1,.5,.9,.99,1]).round(4).to_dict())
    print("common rates:", positive.round(4).value_counts().head(12).to_dict())
    print("dose/rate:", (d.dose/d.rate).round(5).value_counts().head(6).to_dict())

print("\n=== PAIRED SENSORS ===")
for a,b in [("glucose_finger","glucose_lab"),("map_cuff","map_arterial")]:
    paired = pan[[a,b]].dropna()
    dif = paired[a]-paired[b]
    by_stay = dif.groupby(level=0).mean()
    print(a, "-", b, "n",len(paired), "correlation",round(paired.corr().iloc[0,1],4),
          "difference quantiles", dif.quantile([.01,.1,.5,.9,.99]).round(2).to_dict(),
          "SD of stay mean differences",round(by_stay.std(),3))

print("\n=== DOSE POLICY VS MOST RECENT TARGET READING ===")
for drug, reading, edges in [
    ("insulin","glucose_finger",np.arange(60,301,15)),
    ("norepinephrine","map_arterial",np.arange(25,116,5))
]:
    d = pd.DataFrame({"reading":last.loc[valid_hours,reading],
                      "rate":pan.loc[valid_hours,drug]})
    d["range"] = pd.cut(d.reading, edges)
    table = d.groupby("range", observed=True).agg(
        n=("rate","size"), treated=("rate",lambda r:(r>0).mean()),
        mean_rate=("rate","mean"), sd_rate=("rate","std"))
    print(drug)
    print(table.round(3).to_string())

print("\n=== RAW NEXT-HOUR ASSOCIATIONS (NOT CAUSAL ESTIMATES) ===")
for drug, reading in [("insulin","glucose_finger"),("norepinephrine","map_arterial")]:
    ok = valid_hours & pan[reading].notna().to_numpy() & next_obs[reading].notna().to_numpy()
    r = pan.loc[ok,drug].to_numpy()
    y = next_obs.loc[ok,reading].to_numpy()
    now = pan.loc[ok,reading].to_numpy()
    print(drug,"n",ok.sum(),"corr(rate, current)",round(np.corrcoef(r,now)[0,1],4),
          "corr(rate, next)",round(np.corrcoef(r,y)[0,1],4),
          "corr(rate, change)",round(np.corrcoef(r,y-now)[0,1],4))

print("\n=== FORWARD CSV FORMAT ===")
base_roll = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print(base_roll.head(6).to_string(index=False))
print("columns:", base_roll.columns.tolist())
note(f"""## Run {PY_RUNS}; forward requests used: {FWD_RUNS}
Original dose-edit predictions on tuning stays 2020000000–2020000099:
insulin x2 -0.382; insulin half +0.209 mg/dL;
NE x2 +0.103; NE half -0.307 mmHg.
These are predictions, not estimated truth.
Built hourly panels, preserving measurement missingness and filling absent doses
with zero. Readings are at integer hours; infusions start at hour +0.0833.
Next step: identify dose-policy variation and test adjusted outcome effects.
""")
```
