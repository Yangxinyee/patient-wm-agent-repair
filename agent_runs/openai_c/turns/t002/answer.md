The architecture has a separate drug-action encoder, which may allow targeted edits without disturbing measurement prediction. It does not establish either drug’s effect. I’ll inspect timing and dose variation next; an apparent effect must survive adjustment for the condition preceding treatment, rather than merely reflect why the dose was chosen.

```python
PYTHON_RUNS += 1
train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")

print("=== DATA SUMMARY ===")
for label, df in [("train", train), ("tuning", tuning)]:
    print(label, "rows:", len(df), "stays:", df.stay_id.nunique(),
          "hour range:", (df.hour.min(), df.hour.max()))
    print(df.groupby("event").agg(
        rows=("stay_id", "size"),
        distinct_hours=("hour", "nunique"),
        value_mean=("value", "mean"),
        rate_mean=("rate", "mean")
    ).round(4).to_string())

print("\n=== FIRST STAY, CHRONOLOGICAL ===")
first_id = train.stay_id.iloc[0]
print(train.loc[train.stay_id == first_id,
                ["stay_id", "hour", "event", "value", "rate", "dose"]]
      .sort_values("hour", kind="stable").head(45).to_string(index=False))

print("\n=== DRUG NORMALISATION ===")
for item in VOCAB["numeric"]:
    if item["code"] in ["insulin", "norepinephrine"]:
        print(json.dumps(item))

print("\n=== TARGETABLE TENSORS ===")
for name in NAMES:
    if (name.startswith("encode_action.") or
        name.startswith("embed.numeric") or
        name.startswith("embed.bins.") or
        name.startswith("attr_heads.value_bin") or
        name.startswith("embed_transition.value_bin") or
        name.startswith("core.z_") or
        name.startswith("core.out_norm")):
        print(name, SHAPES[name])

print("\n=== HOURS AND SAME-TIME DUPLICATES ===")
print("First 35 distinct hours:", np.sort(train.hour.unique())[:35])
print("Repeated stay/hour/event rows:",
      int(train.duplicated(["stay_id", "hour", "event"]).sum()))
print("Training stay IDs:", train.stay_id.drop_duplicates().head(12).tolist())
print("Tuning stay IDs:", tuning.stay_id.drop_duplicates().head(12).tolist())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Inspection, run 2\n"
        "- First output was truncated after the vocabulary; loaded records and "
        "requested compact summaries.\n"
        "- Architecture separates action encoding from observation encoding. "
        "Candidate edits must still be justified by estimated drug effects.\n"
        "- No weight edits; no forward passes.\n"
    )
```
