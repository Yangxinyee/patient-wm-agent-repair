The architecture shows two dose-input paths: normalized rate/dose numbers and rate/dose bins, both feeding the action encoder. The first output was truncated before the data sample, so no drug effect is established yet. I’ll inspect timing, dose distributions, and relevant tensor shapes; usable dose variation after accounting for prior readings would support identification.

```python
import os, collections, time
t0 = time.time()
train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")
print("Loaded:", train.shape, tuning.shape, "seconds:", round(time.time()-t0, 2))
print("Columns:", train.columns.tolist())
print("Train stays:", train.stay_id.nunique(), "Tuning stays:", tuning.stay_id.nunique())
print("Events:\n", train.groupby("event").agg(
    n=("stay_id","size"), min_hour=("hour","min"), max_hour=("hour","max")
).to_string())
print("\nTiming, most common hours per event:")
for event, g in train.groupby("event"):
    print(event, g.hour.value_counts().head(7).to_dict())

print("\nNumeric normalization:")
for item in VOCAB["numeric"]:
    print(item)
print("\nVocabulary table sizes:")
for key, obj in VOCAB.items():
    if key not in ["numeric", "config", "codes"]:
        print(key, str(obj)[:350])

print("\nRelevant tensors:")
for name in NAMES:
    if any(s in name for s in (
        "embed.numeric", "embed.bins", "attr_heads.value", "last_numeric",
        "encode_action", "embed_transition.value", "z_post", "z_prior",
        "z_head", "query_norm", "type_emb"
    )):
        print(name, SHAPES[name])

ids = train.stay_id.drop_duplicates().tolist()
tune_ids = tuning.stay_id.drop_duplicates().tolist()
print("\nStay ID examples:", ids[:8], tune_ids[:8])
print("\nFirst stay, hours 10–15:")
print(train[(train.stay_id == ids[0]) & train.hour.between(10,15)].to_string(index=False))
print("\nFirst stay, early rows:")
print(train[train.stay_id == ids[0]].head(12).to_string(index=False))
print("\nDrug distributions:")
print(train[train.event.isin(["insulin","norepinephrine"])].groupby("event")[
    ["rate","dose"]
].describe().to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 1\nRead architecture. Dose enters shared numeric embeddings "
            "and rate/dose bin embeddings, then a separate action encoder. "
            "Outcome heads also use last same-code reading registers. "
            "No effect conclusions or edits yet; full printed vocabulary was truncated.\n")
```
