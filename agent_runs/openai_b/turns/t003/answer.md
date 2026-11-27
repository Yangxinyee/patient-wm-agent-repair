Readings occur on the hour, and infusions start five minutes later. Doses are discrete or rounded, so treatment thresholds may help distinguish drug effects from the conditions that prompted treatment. I’ll build hourly transitions and inspect that policy; a dose association that disappears after adjustment would warn against treating it as causal.

```python
from sklearn.linear_model import LinearRegression

measurement_events = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drug_events = ["insulin", "norepinephrine"]

def make_panel(df):
    m = df[df.event.isin(measurement_events)].pivot(
        index=["stay_id", "hour"], columns="event", values="value"
    )
    a = df[df.event.isin(drug_events)].copy()
    a["hour"] = np.floor(a.hour).astype(int)
    a = a.pivot(index=["stay_id", "hour"], columns="event", values="rate")
    idx = pd.MultiIndex.from_product(
        [df.stay_id.unique(), np.arange(-1, 25)], names=["stay_id", "hour"]
    )
    p = m.reindex(idx).join(a).sort_index()
    p[drug_events] = p[drug_events].fillna(0)
    for e in measurement_events + drug_events:
        g = p.groupby(level=0)[e]
        p[e+"_next"] = g.shift(-1)
        p[e+"_lag"] = g.shift(1)
    return p

panel = make_panel(train)
tpanel = make_panel(tuning)
print("Panel shape:", panel.shape)
print("\nMeasurement availability:")
print(panel[measurement_events].notna().mean().round(3).to_string())

for drug, outcome, step in [
    ("insulin", "glucose_finger", 10),
    ("norepinephrine", "map_arterial", 5)
]:
    print("\n===", drug, "===")
    rates = panel.loc[panel[drug] > 0, drug]
    print("Positive rates:", rates.value_counts().sort_index().head(35).to_dict())
    print("Rate quantiles:", rates.quantile([0,.1,.25,.5,.75,.9,1]).to_dict())
    q = panel[[outcome, outcome+"_next", drug]].dropna().copy()
    q["bucket"] = np.floor(q[outcome]/step)*step
    q["change"] = q[outcome+"_next"] - q[outcome]
    tab = q.groupby("bucket").agg(
        n=(drug,"size"), rate_mean=(drug,"mean"),
        rate_sd=(drug,"std"), next_change=("change","mean")
    )
    print("Policy and transitions (buckets with >=500 pairs):")
    print(tab[tab.n >= 500].round(3).to_string())
    y = q[outcome+"_next"].to_numpy()
    r = q[drug].to_numpy()
    x = (q[outcome].to_numpy() - q[outcome].median()) / q[outcome].std()
    naive = LinearRegression().fit(r[:,None], y)
    adjusted = LinearRegression().fit(
        np.column_stack([r, x, x*x, x*x*x]), y
    )
    print("Naive next-reading slope per rate:", round(naive.coef_[0],4))
    print("Current-reading cubic-adjusted slope:", round(adjusted.coef_[0],4),
          "(exploratory, not yet causal)")

print("\nInfusion counts by hour:")
print(panel.loc[panel.index.get_level_values("hour") >= 0, drug_events]
      .gt(0).groupby(level=1).sum().to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 2: records\n"
        "2,057,535 events, 30,000 training stays; 400 tuning stays. "
        "Readings at integer hours -1..24; actions at h+0.0833, end one hour later. "
        "Insulin in 18,508 stays, norepinephrine in 13,200. "
        "Sensors have substantial missingness and different populations. "
        "Run 3 constructs hourly panels and inspects policy/transition "
        "associations; these are explicitly not causal estimates yet.\n"
    )
```