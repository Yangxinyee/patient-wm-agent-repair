The architecture has a separate drug-action encoder, but its recurrent state can still confuse treatment selection with treatment response. The output was truncated, so I’ll inspect the records and the relevant tensors more selectively. A usable effect estimate will require dose variation beyond what prior readings predict.

```python
NUM = {(x["code"], x["attribute"]): x for x in VOCAB["numeric"]}
CODE = {x["name"]: x["id"] for x in VOCAB["codes"]}

for drug in ["insulin", "norepinephrine"]:
    print("=== NORMALISATION:", drug, "===")
    for (code, attribute), spec in NUM.items():
        if code == drug:
            print(attribute, spec)

print("\n=== NON-CORE TENSORS ===")
for name in NAMES:
    if not name.startswith("core."):
        print(name, SHAPES[name])

df = pd.read_csv("/data/train.csv")
tune = pd.read_csv("/data/tuning.csv")
print("\n=== RECORD COUNTS ===")
print("Train:", df.shape, "stays:", df.stay_id.nunique())
print("Tuning:", tune.shape, "stays:", tune.stay_id.nunique())
print(df.groupby("event", observed=True).agg(
    rows=("stay_id", "size"), min_hour=("hour", "min"), max_hour=("hour", "max")
).to_string())
print("Tuning stay IDs:", tune.stay_id.unique()[:100].tolist())

sid0 = df.stay_id.iloc[0]
print("\n=== FIRST STAY ===")
print(df[df.stay_id == sid0].head(48).to_string(index=False))
print("\n=== DRUG RATE DISTRIBUTIONS ===")
print(df[df.event.isin(["insulin", "norepinephrine"])].groupby("event").rate.describe(
    percentiles=[.01, .1, .25, .5, .75, .9, .99]
).to_string())
print("\n=== MEASUREMENT DISTRIBUTIONS ===")
print(df[df.value.notna()].groupby("event").value.describe().to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 2: architecture\n"
            "Drug atoms have their own action encoder, but enter posterior and recurrent updates. "
            "Value heads also use last-reading registers and code-specific bin transitions. "
            "No causal conclusion yet. Full records loaded for timing and identification analysis.\n"
            "Python runs: 2. Forward requests: 0.\n")
```
