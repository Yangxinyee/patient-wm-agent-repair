The architecture confirms that doses affect the recurrent state through a separate action encoder, while reading predictions also use a last-value register. That gives us several possible editing points, but no evidence yet about the drugs’ effects. I’ll inspect treatment timing, dose variation, and the relevant weights before choosing an identification method.

```python
PY_RUNS += 1
cols = ["stay_id", "hour", "event", "value", "rate", "dose", "end_hour"]
train = pd.read_csv("/data/train.csv", usecols=cols, dtype={"event": "category"})
tuning = pd.read_csv("/data/tuning.csv", usecols=cols, dtype={"event": "category"})

print("=== DATA SIZE ===")
for label, df in [("train", train), ("tuning", tuning)]:
    print(label, "rows", len(df), "stays", df.stay_id.nunique(),
          "memory_MB", round(df.memory_usage(deep=True).sum()/1e6, 1))
    print(df.groupby("event", observed=True).agg(
        rows=("stay_id", "size"), first_hour=("hour", "min"),
        last_hour=("hour", "max"), value_mean=("value", "mean"),
        rate_mean=("rate", "mean"), rate_min=("rate", "min"),
        rate_max=("rate", "max")).round(4).to_string())

train_ids = train.stay_id.drop_duplicates().tolist()
tuning_ids = tuning.stay_id.drop_duplicates().tolist()
print("\n=== FIRST TRAIN STAY ===")
print(train[train.stay_id == train_ids[0]].head(65).to_string(index=False))
print("\n=== TUNING IDS FOR FORWARD REQUESTS ===")
print(tuning_ids[:100])

print("\n=== DRUG NORMALISATION ===")
for entry in VOCAB["numeric"]:
    if entry["code"] in ("insulin", "norepinephrine"):
        print(entry)

print("\n=== RELEVANT WEIGHTS ===")
prefixes = ("embed.", "encode_action.", "attr_heads.value_bin",
            "embed_transition.value_bin", "last_numeric.", "last_status.",
            "query_norm.", "type_emb.", "core.z_post.", "core.z_head.",
            "core.z_in.", "core.prelude_norm.", "core.out_norm.")
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== TIMING AND RATE VARIATION ===")
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
              "insulin", "norepinephrine"):
    d = train[train.event == event]
    print(event, "hour fractional parts:",
          np.round(np.mod(d.hour.to_numpy(), 1), 5).tolist()[:12],
          "common:", np.round(np.mod(d.hour.to_numpy(), 1), 5))
    if event in ("insulin", "norepinephrine"):
        print("rate quantiles", d.rate.quantile([0,.01,.1,.25,.5,.75,.9,.99,1]).round(5).to_dict())
        print("most common rates", d.rate.round(5).value_counts().head(12).to_dict())
        print("infusion lengths", (d.end_hour-d.hour).round(5).value_counts().head(5).to_dict())

note(f"""## Run {PY_RUNS}
Architecture: action doses enter encode_action, posterior z, and the recurrent
prelude. Measurement heads additionally use a last-value register.
Loaded training and tuning records. Training stays: {len(train_ids)};
tuning stays: {len(tuning_ids)}. No effect conclusions or weight edits yet.
""")
```
