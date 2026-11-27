The rollout request failed because the tool’s backend file was missing; it returned no model evidence and used no pass. I’ll continue with the records by aligning each dose to the readings before and after it. I’ll also compare measurement types, since systematic differences between them could otherwise look like treatment effects.

```python
measurements = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drugs = ["insulin", "norepinephrine"]

def make_panel(records):
    stays = np.sort(records.stay_id.unique())
    index = pd.MultiIndex.from_product(
        [stays, np.arange(-1, 25)], names=["stay_id", "h"]
    )
    m = records.loc[records.event.isin(measurements),
                    ["stay_id", "hour", "event", "value"]].copy()
    m["h"] = np.rint(m.hour).astype(int)
    p = m.pivot_table(index=["stay_id", "h"], columns="event",
                      values="value", observed=True).reindex(index)
    a = records.loc[records.event.isin(drugs),
                    ["stay_id", "hour", "event", "rate"]].copy()
    a["h"] = np.floor(a.hour).astype(int)
    rates = a.pivot_table(index=["stay_id", "h"], columns="event",
                         values="rate", observed=True).reindex(index).fillna(0)
    p = p.join(rates).astype("float32")
    p.columns.name = None
    return p

P = make_panel(Rtrain)
PT = make_panel(Rtune)
print("=== PANEL ===")
print(P.shape)
print(P.head(30).to_string())
print("\n=== MEASUREMENT AVAILABILITY BY HOUR ===")
print(P[measurements].notna().groupby(level="h").mean().round(3).to_string())

print("\n=== SAME-HOUR MEASUREMENT DIFFERENCES ===")
offsets = {}
for first, second in [("glucose_finger", "glucose_lab"),
                       ("map_arterial", "map_cuff")]:
    pair = P[[first, second]].dropna()
    delta = pair[first] - pair[second]
    offsets[(first, second)] = float(delta.mean())
    print(first, "-", second, "paired n =", len(pair))
    print(delta.describe(percentiles=[.01, .1, .5, .9, .99]).to_string())
    print("By hour:",
          delta.groupby(level="h").mean().round(2).to_dict())

print("\n=== SIMPLE ASSOCIATIONS, NOT CAUSAL ESTIMATES ===")
for drug, target in [("insulin", "glucose_finger"),
                     ("norepinephrine", "map_arterial")]:
    nxt = P.groupby(level="stay_id")[target].shift(-1)
    mask = P[target].notna() & nxt.notna()
    z = pd.DataFrame({"dose": P.loc[mask, drug],
                      "current": P.loc[mask, target],
                      "next": nxt.loc[mask]})
    z["change"] = z["next"] - z["current"]
    z["dose_group"] = pd.qcut(z.dose, 8, duplicates="drop")
    print(drug, z.groupby("dose_group", observed=True).agg(
        n=("dose", "size"), dose=("dose", "mean"),
        current=("current", "mean"), next=("next", "mean"),
        change=("change", "mean")
    ).round(3).to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Forward service issue\nFirst original dose_edit/tuning request failed "
        "because backend cuts_tuning.parquet was missing. It was not counted. "
        "No model rollout evidence yet. Built hourly panels: readings at integer "
        "hours, doses at hour + 0.0833; absent infusion means rate zero.\n"
        "Python runs: 3. Forward passes: 0.\n"
    )
```
