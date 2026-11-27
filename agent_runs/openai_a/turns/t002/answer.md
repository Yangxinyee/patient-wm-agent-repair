The architecture separates drug-action encoding, but both drugs can influence the recurrent state and reading predictions. The output was truncated before the data details, so I can’t estimate effects yet. I’ll inspect those next. Dose variation among comparable histories would support estimating effects; nearly deterministic dosing would limit identification.

```python
print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if (name.startswith(("encode_action", "z_", "last_", "query", "type_", "gap"))
        or name.startswith("embed.numeric")):
        print(name, SHAPES[name])

print("\n=== VOCABULARY ===")
print("Top-level keys:", list(VOCAB))
vocab_summary = []
for key, value in VOCAB.items():
    text = json.dumps(value)
    limit = 1000 if any(s in key.lower() for s in ["norm", "scale", "stat", "code", "config"]) else 220
    vocab_summary.append(key + ": " + text[:limit])
print("\n".join(vocab_summary)[:3500])

dtype = {
    "hour": "float32", "event": "category", "value": "float32",
    "rate": "float32", "dose": "float32", "end_hour": "float32",
    "rate_unit": "category", "dose_unit": "category",
}
train = pd.read_csv("/data/train.csv", dtype=dtype)
tuning = pd.read_csv("/data/tuning.csv", dtype=dtype)

print("\n=== RECORD SUMMARY ===")
for label, frame in [("train", train), ("tuning", tuning)]:
    print(label, "rows:", len(frame), "stays:", frame.stay_id.nunique(),
          "ID examples:", frame.stay_id.unique()[:8].tolist())
    print(frame.event.value_counts().to_string())
    print("Hours:", frame.hour.min(), frame.hour.max())
print("\nFirst 12 training rows:")
print(train.head(12).to_string(index=False))

print("\n=== DRUG DISTRIBUTIONS ===")
for drug in ["insulin", "norepinephrine"]:
    rows = train.loc[train.event == drug]
    print(drug, "rows:", len(rows), "treated stays:", rows.stay_id.nunique())
    print(rows[["hour", "rate", "dose"]].describe(
        percentiles=[.01, .1, .5, .9, .99]).round(4).to_string())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Architecture inspected; Python 2, forward 0\n"
        "- Drug atoms use encode_action, enter the latent posterior and recurrent prelude.\n"
        "- Reading distributions have 17-bin heads plus code/last-bin transition offsets.\n"
        "- Dose edits alter both numeric scores and dose/rate bins; rollouts feed generated readings back.\n"
        "- Need record-derived effects and baseline forward results before choosing edits.\n"
    )
```