The model processes drugs through a separate action encoder, but its value predictions also depend heavily on previous readings. That gives us several possible edit locations, not yet evidence of a drug effect. Next I’ll inspect timing and dose variation in the records; useful comparisons require knowing which readings precede each infusion.

```python
import re, time
t0 = time.time()
train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")
print("Loaded in", round(time.time()-t0, 2), "seconds")
print("Shapes:", train.shape, tuning.shape)
print("Columns:", list(train.columns))
print("Training stays:", train.stay_id.nunique(),
      "Tuning stays:", tuning.stay_id.nunique())
print("Tuning IDs:", tuning.stay_id.unique()[:100].tolist())

print("\n=== ONE TRAINING STAY ===")
sid0 = train.stay_id.iloc[0]
print(train[train.stay_id == sid0].head(42).to_string(index=False))
print("\n=== EVENT SUMMARY ===")
print(train.groupby("event").agg(
    n=("stay_id", "size"), stays=("stay_id", "nunique"),
    min_hour=("hour", "min"), max_hour=("hour", "max"),
    mean_value=("value", "mean"), mean_rate=("rate", "mean")
).to_string())
print("\nHour fractions:",
      np.unique(np.round(train.hour.to_numpy() % 1, 6))[:50])

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "embed.numeric", "embed.bins", "embed.code", "encode_action",
        "last_numeric", "value_bin", "z_post", "z_head", "z_in"
    ]):
        print(name, SHAPES[name])

print("\n=== COMPLETE VALUE CENTRES ===")
print(VOCAB["value_bin_centres"])
with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 1: architecture\n"
        "Separate DeepSets action encoder; action influences posterior latent "
        "and prelude/state. Value head is linear on with_code plus a per-code/"
        "last-bin transition table. Numeric embedding jointly uses value, "
        "dose and rate. Same-code last-reading register is important. "
        "No causal conclusion yet. Initial output was truncated; run 2 "
        "loads records and prints focused diagnostics.\n"
    )
```