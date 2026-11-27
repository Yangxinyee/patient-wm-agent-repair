The architecture separates drug events from measurements, but it does not enforce causal drug effects. Dose information can also enter the latent state, so simply deleting action inputs could damage factual predictions. I’ll inspect timing, dose variation, and measurement coverage next; useful dose variation among similar histories would support identifying effects from the records.

```python
print("=== REMAINING TENSORS ===")
for name in NAMES:
    if name.startswith(("embed.", "encode", "last_", "z_", "query_", "type_", "monitor_", "pending_", "filter_")):
        print(name, SHAPES[name])

print("\n=== VOCAB KEYS ===")
for key, value in VOCAB.items():
    size = len(value) if hasattr(value, "__len__") else "-"
    print(key, type(value).__name__, size)
for key in ["codes", "config", "normalisation", "normalization", "norm", "stats"]:
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key])[:2200])

dtypes = {
    "hour": "float32", "value": "float32", "rate": "float32",
    "dose": "float32", "end_hour": "float32",
    "event": "category", "rate_unit": "category", "dose_unit": "category"
}
train = pd.read_csv("/data/train.csv", dtype=dtypes)
tuning = pd.read_csv("/data/tuning.csv", dtype=dtypes)
print("\n=== RECORD SIZES ===")
print("train:", train.shape, "stays:", train.stay_id.nunique())
print("tuning:", tuning.shape, "stays:", tuning.stay_id.nunique())
print("columns:", train.columns.tolist())
print("\n=== EVENT COUNTS ===")
print(train.event.value_counts().to_string())

sid0 = train.stay_id.iloc[0]
print("\n=== FIRST STAY, FIRST 24 EVENTS ===")
print(train.loc[train.stay_id == sid0].head(24).to_string(index=False))

print("\n=== MEASUREMENT VALUES ===")
print(train.groupby("event", observed=True).value.agg(
    ["count", "mean", "std", "min", "max"]).dropna(subset=["mean"]).to_string())
print("\n=== INFUSION RATES ===")
print(train.groupby("event", observed=True).rate.agg(
    ["count", "mean", "std", "min", "max"]).dropna(subset=["mean"]).to_string())
print("\n=== EVENT TIMING EXAMPLES ===")
for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff", "insulin", "norepinephrine"]:
    x = train.loc[train.event == ev, "hour"]
    print(ev, "range", (float(x.min()), float(x.max())),
          "fractional hours", np.unique(np.round(x.to_numpy() % 1, 5))[:18])

Path("/work/notes.md").open("a").write(
    "\n## Run 2: architecture and record inspection\n"
    "- Actions use a separate set encoder but enter the posterior and recurrent update.\n"
    "- Value predictions also depend on a same-code last-value register.\n"
    "- Need identify dose effects from temporal records before editing; action deletion alone is not justified.\n"
    "- Python runs: 2. Forward passes: 0.\n"
)
```