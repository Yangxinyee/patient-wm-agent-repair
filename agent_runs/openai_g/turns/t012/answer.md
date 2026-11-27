The tensor layout confirms that we can target dose magnitudes without removing measurements or drug presence. Before editing, I’ll collect the drug-specific normalization and dose distributions so replacement inputs can preserve their training averages.

The test will be whether removing magnitude information makes all four rollout shifts zero while keeping factual NLL within tolerance.

```python
import pandas as pd
import numpy as np
import json

print("=== NUMERIC NORMALIZATION ===")
for entry in VOCAB["numeric"]:
    text = json.dumps(entry, default=str)
    if "insulin" in text or "norepinephrine" in text:
        print(text)
    else:
        print("Other entry:", text[:180])

print("\n=== SMALL SURVIVING STRUCTURES ===")
printed_chars = 0
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, (list, tuple, dict)) and len(obj) <= 100:
        text = str(obj)
        if printed_chars + min(len(text), 700) < 3000:
            print(name, text[:700])
            printed_chars += min(len(text), 700)

print("\n=== EXAMPLE RECORDS ===")
print(pd.read_csv("/data/train.csv", nrows=12).to_string(index=False))

drug_code_ids = {
    x["name"]: x["id"] for x in VOCAB["codes"]
    if x["name"] in ("insulin", "norepinephrine")
}
drug_parts = []
observed_units = {name: set() for name in drug_code_ids}
for chunk in pd.read_csv(
    "/data/train.csv",
    usecols=["stay_id", "hour", "event", "rate", "dose", "rate_unit", "dose_unit"],
    dtype={
        "stay_id": "int32", "hour": "float32",
        "rate": "float32", "dose": "float32"
    },
    chunksize=400000,
):
    chunk = chunk.loc[chunk["event"].isin(drug_code_ids)].copy()
    for drug in drug_code_ids:
        rows = chunk.loc[chunk["event"] == drug]
        observed_units[drug].update(
            tuple(x) for x in rows[["rate_unit", "dose_unit"]]
            .drop_duplicates().itertuples(index=False, name=None)
        )
    chunk["code"] = chunk["event"].map(drug_code_ids).astype("uint8")
    drug_parts.append(chunk[["stay_id", "hour", "code", "rate", "dose"]].copy())

drug_records = pd.concat(drug_parts, ignore_index=True)
del drug_parts, chunk

print("\n=== TRAINING DRUG DISTRIBUTIONS ===")
print("Drug records:", len(drug_records))
print("Memory MB:", drug_records.memory_usage(deep=True).sum() / 1e6)
print("Units:", observed_units)
for drug, code in drug_code_ids.items():
    rows = drug_records.loc[drug_records["code"] == code]
    print("\n", drug, "rows:", len(rows),
          "missing:", rows[["rate", "dose"]].isna().sum().to_dict())
    print(rows[["rate", "dose"]].describe(
        percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).to_string())

print("\nCurrent edits:", edited())
```