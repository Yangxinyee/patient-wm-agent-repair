The architecture shows that doses affect the recurrent state through action embeddings, while measurement predictions also use the last recorded value. The records—not the architecture—must establish the drug effects. I’ll inspect timing, dose variation, and relevant weights; useful dose variation among similar patient histories would support estimating an effect.

```python
train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")

print("=== DATA SHAPE ===")
for label, df in [("train", train), ("tuning", tuning)]:
    print(label, df.shape, "stays:", df.stay_id.nunique())
    print("first stay ids:", df.stay_id.drop_duplicates().head(8).tolist())
print("columns:", train.columns.tolist())
print("event counts:\n", train.event.value_counts().to_string())
print("hour range:", train.hour.min(), train.hour.max())
print("noninteger-hour fraction:", np.mean(np.abs(train.hour - train.hour.round()) > 1e-6))
print("events per stay:", train.groupby("stay_id").size().describe().round(2).to_dict())

print("\n=== FIRST STAY, HOURS -1 THROUGH 8 ===")
first_id = train.stay_id.iloc[0]
print(train[(train.stay_id == first_id) & (train.hour <= 8)]
      [["stay_id", "hour", "event", "value", "rate", "dose", "end_hour"]]
      .to_string(index=False))

print("\n=== MEASUREMENT AND DOSE DISTRIBUTIONS ===")
for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
              "insulin", "norepinephrine"]:
    sub = train[train.event == event]
    col = "rate" if event in ["insulin", "norepinephrine"] else "value"
    print(event, "n =", len(sub), "hours =", sub.hour.value_counts().sort_index().to_dict())
    print(sub[col].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99]).round(5).to_dict())
    if col == "rate":
        print("rate-dose ratio:", (sub.dose / sub.rate.replace(0, np.nan))
              .describe().round(6).to_dict())
        print("most common rates:", sub.rate.value_counts().head(8).to_dict())

print("\n=== RELEVANT TENSORS ===")
prefixes = ("embed.", "encode_action.", "attr_heads.value", "embed_transition.value",
            "last_numeric", "last_status", "query_norm", "type_emb", "core.z_post")
for name in NAMES:
    if name.startswith(prefixes):
        w = load_original(name)
        print(name, SHAPES[name], "rms =", round(float(np.sqrt(np.mean(w*w))), 5))

print("\n=== VOCAB DETAILS ===")
print("keys:", list(VOCAB))
for entry in VOCAB["numeric"]:
    if entry["code"] in ["insulin", "norepinephrine"]:
        print(entry)
for key in VOCAB:
    if key not in ["numeric", "codes", "event_kinds", "atom_kinds", "config"]:
        obj = VOCAB[key]
        print(key, str(obj)[:1200])

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython run 2. Architecture: action embeddings enter posterior and recurrent "
            "state; value heads also have same-code last-value registers and bin transitions. "
            "Need establish causal dose response using temporal records before editing.\n")
```
