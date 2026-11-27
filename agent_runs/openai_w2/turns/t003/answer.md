Infusions start five minutes after hourly readings, which lets us distinguish pre-dose measurements from later outcomes. The two glucose channels and two MAP channels may also help control for patient condition and measurement noise. I’ll examine dosing rules and reading dynamics next. Consistent effects after controlling for pre-dose history would support a causal estimate; unstable or reversed estimates would suggest residual confounding.

```python
session["python_runs"] += 1

print("=== NUMERIC ENCODING ===")
print(json.dumps(VOCAB["numeric"], indent=2))
for name in NAMES:
    if name.startswith(("embed.numeric", "embed_transition", "last_numeric")):
        print(name, SHAPES[name])

measurement_events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drug_events = ["insulin", "norepinephrine"]

def make_panel(df):
    x = df[df.event.isin(measurement_events + drug_events)].copy()
    x["t"] = np.floor(x.hour + 1e-3).astype(np.int16)
    x["x"] = x["value"].where(x.event.isin(measurement_events), x["rate"])
    assert not x.duplicated(["stay_id", "t", "event"]).any()
    p = x.pivot(index=["stay_id", "t"], columns="event", values="x")
    idx = pd.MultiIndex.from_product(
        [np.sort(df.stay_id.unique()), np.arange(-1, 25)],
        names=["stay_id", "t"]
    )
    p = p.reindex(idx).sort_index()
    p[drug_events] = p[drug_events].fillna(0)
    return p

panel = make_panel(train)
tune_panel = make_panel(tuning)
print("\n=== HOURLY PANEL ===")
print(panel.shape)
print("observation fractions:", panel[measurement_events].notna().mean().round(3).to_dict())

print("\n=== PAIRED CHANNEL DIFFERENCES ===")
for a, b in [("glucose_finger", "glucose_lab"), ("map_arterial", "map_cuff")]:
    d = panel[a] - panel[b]
    means = d.groupby(level="stay_id").mean()
    within = d - d.groupby(level="stay_id").transform("mean")
    print(a, "-", b, {
        "n_pairs": int(d.notna().sum()),
        "mean": round(d.mean(), 3),
        "sd": round(d.std(), 3),
        "between_stay_sd": round(means.std(), 3),
        "within_stay_sd": round(within.std(), 3),
        "quantiles": d.quantile([.01, .1, .5, .9, .99]).round(2).tolist()
    })

print("\n=== DOSING BY CURRENT READING ===")
for drug, obs, edges in [
    ("insulin", "glucose_finger", np.arange(60, 301, 20)),
    ("norepinephrine", "map_arterial", np.arange(30, 111, 5))
]:
    q = panel.loc[panel.index.get_level_values("t") >= 0, [drug, obs]].dropna()
    q["band"] = pd.cut(q[obs], edges)
    tab = q.groupby("band", observed=True).agg(
        n=(drug, "size"), mean_rate=(drug, "mean"),
        active_fraction=(drug, lambda x: (x > 0).mean()),
        rate_sd=(drug, "std")
    )
    print(drug)
    print(tab.round(3).to_string())

print("\n=== SIMPLE TEMPORAL ASSOCIATIONS ===")
for drug, obs in [("insulin", "glucose_finger"), ("norepinephrine", "map_arterial")]:
    nxt = panel.groupby(level="stay_id")[obs].shift(-1)
    q = pd.DataFrame({"dose": panel[drug], "now": panel[obs], "next": nxt}).dropna()
    q["change"] = q["next"] - q["now"]
    print(drug, "n=", len(q), "correlations:",
          q[["dose", "now", "next", "change"]].corr().loc["dose"].round(4).to_dict())

fwd_ids = [int(x) for x in np.sort(tuning.stay_id.unique())[:100]]
print("\n=== FORWARD COHORT (100 TUNING STAYS) ===")
print(fwd_ids)
print("COUNTS:", session)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 3\n"
        "Constructed complete hourly panels (-1 through 24), aligning infusions "
        "at hour+0.0833 to that hour's pre-dose measurements. Missing infusion "
        "records mean zero exposure. Auditing paired measurement channels, "
        "dose-assignment rules, and next-hour associations before estimating "
        "causal effects. No edits yet.\n"
    )
```