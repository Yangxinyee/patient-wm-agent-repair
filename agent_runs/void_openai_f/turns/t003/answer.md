The architecture separates drug actions from measurements, so an action-path edit may be possible. It also feeds sampled readings back into later predictions, making a rollout check essential. I’ll inspect dose ranges, timing, and the relevant tensor shapes next. Adequate dose variation at similar pre-dose conditions would support effect estimation; near-deterministic dosing would weaken it.

```python
Rtrain = pd.read_csv("/data/train.csv")
Rtune = pd.read_csv("/data/tuning.csv")
for r in (Rtrain, Rtune):
    for c in ("event", "rate_unit", "dose_unit"):
        if c in r:
            r[c] = r[c].astype("category")

print("=== DATA SUMMARY ===")
print("Shapes:", Rtrain.shape, Rtune.shape)
print("Columns:", list(Rtrain.columns))
print("Training stays:", Rtrain.stay_id.nunique())
print("Tuning stays:", Rtune.stay_id.nunique())
print(Rtrain.groupby("event", observed=True).agg(
    rows=("stay_id", "size"), hours_min=("hour", "min"),
    hours_max=("hour", "max"), value_mean=("value", "mean"),
    rate_mean=("rate", "mean")
).to_string())

print("\n=== FIRST STAY: EVENT ORDER ===")
first_id = Rtrain.stay_id.iloc[0]
print(Rtrain.loc[Rtrain.stay_id == first_id,
                 ["stay_id", "hour", "event", "value", "rate", "dose"]]
      .head(48).to_string(index=False))

print("\n=== DOSE NORMALISATION AND DISTRIBUTIONS ===")
for entry in VOCAB["numeric"]:
    if entry["code"] in ("insulin", "norepinephrine"):
        print(json.dumps(entry))
for drug in ("insulin", "norepinephrine"):
    x = Rtrain.loc[Rtrain.event == drug, ["hour", "rate", "dose"]]
    print(drug, x.describe(percentiles=[.01, .1, .5, .9, .99]).to_string())
    print("Distinct rate count:", x.rate.nunique())
    print("Fractional hours:", np.unique(np.round(x.hour % 1, 6))[:30])

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in (
        "embed.", "encode_action.", "attr_heads.value", "last_numeric",
        "embed_transition.value", "core.z_", "query_norm", "type_emb",
        "last_status", "monitor_head"
    )):
        print(name, SHAPES[name])

tuning_ids = np.sort(Rtune.stay_id.unique())
probe_ids = tuning_ids[np.linspace(0, len(tuning_ids)-1, 100).astype(int)].tolist()
print("\nFORWARD PROBE IDS:", json.dumps(probe_ids))
print("Vocabulary top-level keys:", list(VOCAB))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Inspection\nArchitecture inspected. Actions use a separate encoder, "
        "but also enter the latent posterior and recurrent state. Value predictions "
        "combine a state head with same-code last-value registers and transition logits. "
        "Dose-edit rollouts feed sampled readings back. No drug-effect conclusion yet.\n"
        "Python runs: 2. Forward passes: 0.\n"
    )
```
