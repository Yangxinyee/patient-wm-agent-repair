The architecture separates drug inputs from measurement inputs, so there may be a targeted way to repair dose responses. That does not yet tell us what either drug does. I’ll now inspect the complete records and vocabulary, focusing on treatment timing and dose variation; useful variation after comparable histories would support effect estimation, while near-deterministic dosing would limit it.

```python
run_count += 1

print("=== REMAINING TENSORS ===")
for name in NAMES:
    if name.startswith(("encode", "filter", "gap", "last", "monitor",
                        "pending", "query", "time", "type", "z_", "embed.numeric",
                        "embed_transition")):
        print(name, SHAPES[name])

print("\n=== VOCAB KEYS / CONFIG / CODES ===")
print(list(VOCAB))
for key in ("config", "codes", "normalisation", "normalization", "norm", "norms"):
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key], indent=2))
for key, val in VOCAB.items():
    if key not in ("config", "codes"):
        if isinstance(val, dict):
            print(key, "keys:", list(val)[:20])
        else:
            print(key, "type:", type(val).__name__,
                  "length:", len(val) if hasattr(val, "__len__") else None)

df = pd.read_csv(
    "/data/train.csv",
    dtype={"event": "category", "rate_unit": "category", "dose_unit": "category"}
)
tune = pd.read_csv(
    "/data/tuning.csv",
    dtype={"event": "category", "rate_unit": "category", "dose_unit": "category"}
)
print("\n=== DATA SIZE / SAMPLE ===")
print("train", df.shape, "tuning", tune.shape)
print(df.head(14).to_string(index=False))
print(df.dtypes.to_string())

print("\n=== EVENT DISTRIBUTIONS ===")
print(df.groupby("event", observed=True).agg(
    rows=("stay_id", "size"), stays=("stay_id", "nunique"),
    min_hour=("hour", "min"), max_hour=("hour", "max"),
    value_mean=("value", "mean"), value_sd=("value", "std"),
    rate_mean=("rate", "mean"), rate_sd=("rate", "std")
).round(5).to_string())

print("\n=== TIMING AND DOSE SAMPLES ===")
for ev in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
           "insulin", "norepinephrine"):
    sub = df.loc[df.event == ev]
    print(ev, "hours:", sub.hour.value_counts().sort_index().head(28).to_dict())
    if ev in ("insulin", "norepinephrine"):
        print("rate quantiles:", sub.rate.quantile(
            [0, .01, .1, .25, .5, .75, .9, .99, 1]).to_dict())
        print("common rates:", sub.rate.value_counts().head(12).to_dict())
        print("dose/rate:", (sub.dose / sub.rate).describe().round(5).to_dict())

print("\n=== FIRST STAY ===")
first_id = df.stay_id.iloc[0]
print(df.loc[df.stay_id == first_id].head(45).to_string(index=False))
print("tuning IDs:", tune.stay_id.unique()[:100].tolist())

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 2: records and vocabulary\n"
        "- Architecture has a dedicated action encoder; this is a candidate edit boundary, not evidence of drug effects.\n"
        "- Loaded all training and tuning records and inspected event timing and dose distributions.\n"
        f"- Python runs: {run_count}; forward passes: {forward_count}.\n"
    )
```
