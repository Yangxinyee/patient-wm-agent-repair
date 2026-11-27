The architecture shows that doses enter through a separate action encoder, but can then affect every prediction through the recurrent state. The output was truncated before the record samples. I’ll inspect the timing, dose variation, and relevant tensors next; those will determine whether a targeted edit is feasible.

```python
RUNS += 1
train = pd.read_csv(
    "/data/train.csv",
    dtype={"event": "category", "hour": "float32", "value": "float32",
           "rate": "float32", "dose": "float32", "end_hour": "float32",
           "rate_unit": "category", "dose_unit": "category"}
)
tuning = pd.read_csv("/data/tuning.csv")
print("Columns:", train.columns.tolist())
print("Shape:", train.shape, "Stays:", train.stay_id.nunique(),
      "Memory MB:", round(train.memory_usage(deep=True).sum()/1e6, 1))
print("\nEvent summaries:")
print(train.groupby("event", observed=True).agg(
    n=("stay_id", "size"), stays=("stay_id", "nunique"),
    min_hour=("hour", "min"), max_hour=("hour", "max"),
    mean_value=("value", "mean"), mean_rate=("rate", "mean")
).to_string())
print("\nFirst stay, first 22 events:")
first_id = train.stay_id.iloc[0]
print(train[train.stay_id == first_id].head(22).to_string(index=False))
print("\nDistinct hours, first 45:", np.sort(train.hour.unique())[:45])
print("\nDuplicate event/stay/hour rows:",
      train.duplicated(["stay_id", "hour", "event"]).sum())
print("\nDrug rate summaries:")
print(train[train.event.isin(["insulin", "norepinephrine"])].groupby(
    "event", observed=True
)["rate"].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]).to_string())
print("\nVOCAB keys:", list(VOCAB))
print("Drug normalizations:")
for item in VOCAB["numeric"]:
    if item["code"] in ["insulin", "norepinephrine"]:
        print(json.dumps(item))
print("\nRelevant tensor names:")
for name in NAMES:
    if (name.startswith(("embed.", "encode_action.", "attr_heads.value",
                         "embed_transition.value", "last_numeric", "last_status",
                         "query_norm", "type_emb", "core.z_", "z_head"))
        or "filter" in name):
        print(name, SHAPES[name])
print("\nOther vocab table shapes:")
for key, value in VOCAB.items():
    if key not in ["numeric", "config", "codes", "event_kinds", "atom_kinds"]:
        if isinstance(value, dict):
            print(key, "keys:", list(value)[:16])
        else:
            print(key, "shape:", np.shape(value))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 1\nRead full architecture and schema. "
            "Action embeddings feed a shared recurrent state; last-value registers "
            "also feed value heads. No effect estimates or edits yet. "
            "Initial printed output was truncated; run 2 targets missing details.\n")
print("\nCounts:", RUNS, "Python runs;", FORWARDS, "forward passes.")
```