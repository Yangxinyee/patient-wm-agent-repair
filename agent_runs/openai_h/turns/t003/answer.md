Readings occur on the hour, and infusions start five minutes later. That ordering lets us compare next-hour outcomes after conditioning on information available before dosing. I’ll build an hourly panel and inspect paired measurements and dose selection; this will reveal whether apparent drug effects could instead reflect noisy readings or patient trends.

```python
MEAS = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
DRUGS = ["insulin", "norepinephrine"]

def make_panel(df):
    m = df.loc[df.event.isin(MEAS), ["stay_id", "hour", "event", "value"]]
    m = m.pivot_table(index=["stay_id", "hour"], columns="event",
                      values="value", aggfunc="mean", observed=True)
    d = df.loc[df.event.isin(DRUGS), ["stay_id", "hour", "event", "rate"]].copy()
    d["hour"] = np.floor(d.hour).astype("int16")
    d = d.pivot_table(index=["stay_id", "hour"], columns="event",
                      values="rate", aggfunc="mean", observed=True)
    idx = pd.MultiIndex.from_product(
        [np.sort(df.stay_id.unique()), np.arange(-1, 25)],
        names=["stay_id", "hour"]
    )
    p = pd.concat([m, d], axis=1).reindex(idx).astype("float32")
    p[DRUGS] = p[DRUGS].fillna(0)
    p["glucose"] = p[["glucose_finger", "glucose_lab"]].mean(axis=1)
    p["map"] = p[["map_arterial", "map_cuff"]].mean(axis=1)
    return p

panel = make_panel(train)
panel_tune = make_panel(tuning)

print("=== NORMALIZATION AND BIN DEFINITIONS ===")
print("numeric_note:", VOCAB.get("numeric_note"))
for row in VOCAB["numeric"]:
    print(json.dumps(row)[:1050])
print("event_kinds:", VOCAB["event_kinds"])
print("atom_kinds:", VOCAB["atom_kinds"])

print("\n=== PAIRED SENSOR DIFFERENCES ===")
for a, b in [("glucose_finger", "glucose_lab"), ("map_arterial", "map_cuff")]:
    x = panel[[a, b]].dropna()
    delta = x[a] - x[b]
    print(a, "minus", b, "n", len(x), "mean", delta.mean(),
          "sd", delta.std(), "correlation", x.corr().iloc[0, 1])

print("\n=== MEASUREMENT COVERAGE AND DOSING BY HOUR ===")
print(panel.groupby("hour")[MEAS + DRUGS].agg(
    {**{c: "count" for c in MEAS}, **{c: "mean" for c in DRUGS}}
).to_string())

print("\n=== MOST COMMON POSITIVE RATES ===")
for drug in DRUGS:
    print(drug, train.loc[train.event == drug, "rate"].value_counts().head(12).to_dict())

print("\n=== DOSE SELECTION AND NEXT-HOUR CHANGE ===")
for drug, outcome in [("insulin", "glucose"), ("norepinephrine", "map")]:
    y = panel[outcome]
    tmp = pd.DataFrame({
        "dose": panel[drug],
        "current": y,
        "next": y.groupby(level=0).shift(-1),
        "previous": y.groupby(level=0).shift(1)
    }).dropna()
    tmp["change"] = tmp["next"] - tmp["current"]
    tmp["previous_change"] = tmp["current"] - tmp["previous"]
    tmp["dose_group"] = pd.qcut(tmp["dose"], 8, duplicates="drop")
    print(drug)
    print(tmp.groupby("dose_group", observed=True).agg(
        n=("dose", "size"), dose=("dose", "mean"), current=("current", "mean"),
        change=("change", "mean"), previous_change=("previous_change", "mean")
    ).to_string())

EVAL_IDS = [int(x) for x in np.sort(tuning.stay_id.unique())[:100]]
print("\n=== FIRST FORWARD COHORT ===")
print(json.dumps(EVAL_IDS))

Path("/work/notes.md").open("a").write(
    "\n## Run 3: hourly panel\n"
    "- Measurements are at integer hours; drugs start at hour + 0.0833 and run one hour.\n"
    "- Built complete stay-hour panels, with absent infusion records represented as zero rates.\n"
    "- Inspecting paired sensors and pretreatment trends before estimating dose effects.\n"
    "- Python runs: 3. Forward passes: 0.\n"
)
```