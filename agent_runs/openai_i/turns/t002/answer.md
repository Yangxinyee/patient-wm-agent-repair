The architecture separates drug actions from measurements, but dose information also passes through shared embeddings and recurrent state. This identifies possible edit sites—not drug effects. The output was truncated, so I’ll inspect the data and remaining metadata compactly. Dose variation among similar histories would support effect estimation; nearly deterministic dosing would limit it.

```python
RUN_COUNTS["python"] += 1

print("=== VOCAB KEYS ===")
print(list(VOCAB))
for key in ("config", "codes"):
    print(key, json.dumps(VOCAB.get(key), default=str))
print("Other vocabulary structures:")
for key, val in VOCAB.items():
    if key not in ("config", "codes"):
        if isinstance(val, dict):
            print(key, "keys:", list(val)[:25])
        else:
            print(key, type(val).__name__,
                  "length:", len(val) if hasattr(val, "__len__") else "-")

print("\n=== SELECTED REMAINING TENSORS ===")
prefixes = ("embed.numeric", "encode_action.", "last_numeric.",
            "query_norm.", "type_emb", "z_head.", "z_post.", "z_prior.",
            "filter_", "pending_")
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")
for df in (train, tuning):
    for col in ("event", "rate_unit", "dose_unit"):
        if col in df:
            df[col] = df[col].astype("category")

print("\n=== DATA SIZE / EVENTS ===")
print("Train:", train.shape, "stays:", train.stay_id.nunique())
print("Tuning:", tuning.shape, "stays:", tuning.stay_id.nunique())
print("Columns:", list(train))
print(train.groupby("event", observed=True).agg(
    rows=("stay_id", "size"),
    stays=("stay_id", "nunique"),
    min_hour=("hour", "min"),
    max_hour=("hour", "max")
).to_string())

print("\n=== MEASUREMENT AND RATE SUPPORT ===")
for event in train.event.unique():
    sub = train.loc[train.event == event]
    col = "rate" if event in ("insulin", "norepinephrine") else "value"
    vals = sub[col].dropna()
    if len(vals):
        print(event, col, "n=", len(vals), "unique=", vals.nunique(),
              "quantiles=", vals.quantile(
                  [0, .01, .1, .5, .9, .99, 1]
              ).round(5).to_dict())

sid = train.stay_id.iloc[0]
print("\n=== FIRST STAY, FIRST 25 EVENTS ===")
print(train.loc[train.stay_id == sid].sort_values("hour").head(25)
      .to_string(index=False))
print("\nRows per stay:")
print(train.groupby("stay_id", observed=True).size()
      .quantile([0, .1, .5, .9, 1]).to_dict())

rng = np.random.default_rng(20261110)
tune_ids = rng.permutation(tuning.stay_id.unique()).tolist()
tune_dev = tune_ids[:100]
tune_check = tune_ids[100:200]
tune_holdout = tune_ids[200:]

with (work / "notes.md").open("a") as f:
    f.write(
        "\n## Inspection — Python 2; forward 0\n"
        "- Architecture read: separate action encoder; recurrent action pathways; "
        "shared attribute embeddings and same-code last-value registers.\n"
        "- Initial output truncated before vocabulary and sample rows; "
        "reprinted compact metadata and loaded records.\n"
        "- Reserved fixed tuning subsets for development, independent checking, "
        "and final validation. No weights changed.\n"
    )
print("\nCounts:", RUN_COUNTS)
```
