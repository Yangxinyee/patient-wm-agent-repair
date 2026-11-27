The records place doses five minutes after hourly readings, and both drugs have substantial dose variation. That timing lets us compare subsequent changes after controlling for information available before treatment. I’ll build an hourly panel and check whether measurement modalities can be combined; strong modality differences would require separate analyses.

```python
print("=== DRUG RATE NORMALIZATION ===")
for x in VOCAB["numeric"]:
    if x["attribute"] == "rate":
        print(x)
print("=== BIN CENTRES / OTHER VOCAB KEYS ===")
for key in VOCAB:
    if key not in ["numeric", "codes", "config", "atom_kinds", "event_kinds"]:
        text = repr(VOCAB[key])
        print(key, text[:2400])

print("\n=== NON-TRANSFORMER TENSORS ===")
for name in NAMES:
    if not any(x in name for x in ["core.prelude.", "core.blocks.", "core.coda."]):
        print(name, SHAPES[name])

def make_panel(df):
    x = df.loc[df.event.isin([
        "glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
        "insulin", "norepinephrine"
    ]), ["stay_id", "hour", "event", "value", "rate"]].copy()
    x["t"] = np.floor(x.hour + 1e-6).astype(np.int16)
    x["x"] = x.value.where(x.value.notna(), x.rate)
    p = x.pivot(index=["stay_id", "t"], columns="event", values="x")
    idx = pd.MultiIndex.from_product(
        [np.sort(df.stay_id.unique()), np.arange(-1, 25)], names=["stay_id", "t"]
    )
    p = p.reindex(idx)
    for drug in ["insulin", "norepinephrine"]:
        p[drug] = p[drug].fillna(0)
    return p

panel = make_panel(train)
panel_tune = make_panel(tuning)
print("\n=== HOURLY COVERAGE ===")
print(panel.notna().mean().round(4).to_dict())
for a, b in [("glucose_finger", "glucose_lab"), ("map_arterial", "map_cuff")]:
    ab = panel[[a, b]].dropna()
    print(a, b, "simultaneous", len(ab))
    if len(ab):
        print("difference quantiles:", (ab[a]-ab[b]).quantile([.01,.1,.5,.9,.99]).round(3).to_dict(),
              "correlation:", round(ab.corr().iloc[0,1], 4))
    print("per-stay measurement counts:",
          panel[[a,b]].notna().groupby(level=0).sum().quantile([0,.25,.5,.75,1]).to_dict())
print("Drug exposure:")
for drug in ["insulin", "norepinephrine"]:
    v = panel.loc[panel.index.get_level_values("t") >= 0, drug]
    print(drug, "nonzero fraction", round((v>0).mean(),4),
          "mean positive", round(v[v>0].mean(),4),
          "mean all", round(v.mean(),4))
print("tuning stay ids:", tuning.stay_id.unique()[:100].tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Record inspection (runs 2–3)\n"
        "- 30,000 training stays; 400 tuning stays; approximately 2 million training events.\n"
        "- Readings occur at integer hours; one-hour infusion records start at hour + 0.0833.\n"
        "- Insulin rate equals its dose; norepinephrine dose equals 4.8 times its rate.\n"
        "- Insulin rates are half-unit increments; norepinephrine rates are hundredth increments, capped at 1.\n"
        "- Constructed hourly panels `panel` and `panel_tune`; absent infusion means zero exposure.\n"
        "- Need control for treatment assignment, prior readings, patient heterogeneity, and modality selection.\n"
        "- No edits or forward requests yet.\n"
    )
```
