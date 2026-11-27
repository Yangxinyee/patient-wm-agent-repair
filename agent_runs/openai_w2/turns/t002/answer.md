The architecture gives us a distinct drug-action pathway, but it does not establish whether either drug affects subsequent readings. The output was truncated, so I’ll inspect the remaining essentials and load the records. Dose variation beyond what prior readings predict would support estimating effects; near-deterministic dosing would make that harder.

```python
session["python_runs"] += 1

print("=== VOCAB KEYS ===")
for k, v in VOCAB.items():
    detail = list(v.keys()) if isinstance(v, dict) else (
        f"length={len(v)}" if isinstance(v, list) else v
    )
    print(k, detail)

print("\n=== CODES AND CONFIG ===")
print("codes:", VOCAB["codes"])
print("config:", VOCAB["config"])

print("\n=== SELECTED REMAINING TENSORS ===")
for name in NAMES:
    if name.startswith(("encode.", "encode_action.", "z_", "last_", "gap.", "type_emb", "query_norm")):
        print(name, SHAPES[name])

train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")
print("\n=== DATA SIZE ===")
print("train:", train.shape, "stays:", train.stay_id.nunique())
print("tuning:", tuning.shape, "stays:", tuning.stay_id.nunique())
print("columns:", list(train.columns))
print("training event counts:\n", train.event.value_counts().to_string())

print("\n=== SAMPLE ROWS ===")
print(train.head(12).to_string(index=False))

print("\n=== EVENT RANGES ===")
for event, g in train.groupby("event", sort=False):
    fields = {}
    for col in ["hour", "value", "rate", "dose"]:
        x = g[col].dropna()
        if len(x):
            fields[col] = np.round(x.quantile([0, .1, .5, .9, 1]).values, 4).tolist()
    print(event, fields)

first_id = train.stay_id.iloc[0]
first = train[train.stay_id == first_id]
print("\n=== FIRST STAY TIMING ===")
for event, g in first.groupby("event", sort=False):
    print(event, "hours:", np.round(g.hour.values[:30], 4).tolist())
print("\nCURRENT EDITS:", edited())
print("COUNTS:", session)

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Run 2\n"
        "Architecture has separate observation and action encoders, with actions "
        "entering the latent posterior and recurrent prelude. Attribute heads "
        "also use last-value registers. This offers targeted editing routes, "
        "but no drug-effect conclusion has been drawn. Loaded both record files "
        "and inspected schema, vocabulary, distributions, and event timing.\n"
    )
```