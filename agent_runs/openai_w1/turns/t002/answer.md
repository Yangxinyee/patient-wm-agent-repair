The architecture has separate action inputs, but doses also enter through the latent-state update and the last-value registers. Editing only one route may therefore leave dose sensitivity elsewhere. The output was truncated before the record samples; I’ll inspect those and summarize dose timing and variation. Within-history dose variation would support estimating effects from the records; nearly deterministic assignment would make that harder.

```python
python_runs += 1

print("=== COMPACT VOCAB SUMMARY ===")
print("Top-level keys:", list(VOCAB))
for item in VOCAB["numeric"]:
    if item["code"] in ("insulin", "norepinephrine"):
        print(item)
for key in VOCAB:
    if key not in ("numeric", "codes", "config"):
        obj = VOCAB[key]
        print(key, "type:", type(obj).__name__,
              "keys/length:", list(obj)[:12] if isinstance(obj, dict) else len(obj))

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in (
        "embed.numeric", "embed.bins", "encode_action", "last_numeric",
        "value_bin", "z_post", "z_head", "query_norm"
    )):
        print(name, SHAPES[name])

train = pd.read_csv(
    "/data/train.csv",
    dtype={"event": "category", "rate_unit": "category", "dose_unit": "category"}
)
tuning = pd.read_csv(
    "/data/tuning.csv",
    dtype={"event": "category", "rate_unit": "category", "dose_unit": "category"}
)
print("\n=== RECORD STRUCTURE ===")
print("Train:", train.shape, "stays:", train.stay_id.nunique())
print("Tuning:", tuning.shape, "stays:", tuning.stay_id.nunique())
print("Columns:", train.columns.tolist())
print(train.dtypes.to_string())
print(train.groupby("event", observed=True).agg(
    rows=("stay_id", "size"), stays=("stay_id", "nunique"),
    min_hour=("hour", "min"), max_hour=("hour", "max")
).to_string())

first_stay = train.stay_id.iloc[0]
print("\n=== FIRST STAY, EARLY EVENTS ===")
print(train[(train.stay_id == first_stay) & (train.hour <= 3)]
      .head(40).to_string(index=False))

print("\n=== DOSE DISTRIBUTIONS ===")
for drug in ("insulin", "norepinephrine"):
    sub = train[train.event == drug]
    print(drug)
    print(sub[["hour", "rate", "dose"]].describe(
        percentiles=[.01, .1, .25, .5, .75, .9, .99]
    ).round(5).to_string())
    print("Unique rates:", sub.rate.nunique(),
          "zero-rate fraction:", float((sub.rate == 0).mean()))
    print("First distinct hours:", np.sort(sub.hour.unique())[:20])

with notes_path.open("a") as f:
    f.write(
        "\n## Run 2\n"
        "Architecture: doses enter action embeddings, z posterior, state attention, "
        "and last-rate registers. Rollouts feed generated measurements back into state. "
        "Loaded both record splits and inspected timing/dose variation. "
        "Run 1 output was truncated, so later inspection uses compact summaries.\n"
        f"Counts: {python_runs} Python runs, {forward_passes} forward passes.\n"
    )
```
