The model sends drug actions through a shared recurrent state, so an indiscriminate edit could damage every output. The previous output was truncated before the record samples. I’ll inspect their timing, dose variation, and normalisation next; these will determine whether conditional dose effects can be estimated reliably.

```python
print("=== VOCAB ===")
for key, value in VOCAB.items():
    text = json.dumps(value)
    if len(text) <= 1800:
        print(key, text)
    elif isinstance(value, dict):
        print(key, "keys:", list(value))
        for subkey, subvalue in list(value.items())[:3]:
            print(" ", subkey, str(subvalue)[:350])
    else:
        print(key, text[:500], "...")

print("\n=== REMAINING RELEVANT TENSORS ===")
for name in NAMES:
    if name.startswith(("encode", "last", "z_", "query", "type", "embed.numeric", "embed_transition", "gap")):
        print(name, SHAPES[name])

print("\n=== RECORD SAMPLES ===")
cols = ["stay_id", "hour", "event", "value", "rate", "dose", "end_hour"]
print(pd.read_csv("/data/train.csv", nrows=32)[cols].to_string(index=False))
print("\nTuning:")
print(pd.read_csv("/data/tuning.csv", nrows=8)[cols].to_string(index=False))

print("\n=== LOADING RECORDS ===")
train = pd.read_csv("/data/train.csv", usecols=cols)
tuning = pd.read_csv("/data/tuning.csv", usecols=cols)
train["event"] = train["event"].astype("category")
tuning["event"] = tuning["event"].astype("category")
print("Train:", train.shape, "stays:", train.stay_id.nunique())
print("Tuning:", tuning.shape, "stays:", tuning.stay_id.nunique())
print(train.groupby("event", observed=True).agg(
    rows=("stay_id", "size"),
    hour_min=("hour", "min"),
    hour_max=("hour", "max"),
    value_mean=("value", "mean"),
    value_sd=("value", "std"),
    rate_mean=("rate", "mean"),
    rate_sd=("rate", "std"),
).to_string())

print("Tuning stay IDs:", tuning.stay_id.drop_duplicates().head(100).tolist())
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Inspection\n"
        "- Architecture: drug action embeddings affect the recurrent state shared by all heads.\n"
        "- Value heads also have code-specific last-reading transitions.\n"
        "- Broad action edits may affect all-output log loss; prefer selective, evidence-based edits.\n"
        "- Python runs: 2. Forward passes: 0. No edits yet.\n"
    )
```