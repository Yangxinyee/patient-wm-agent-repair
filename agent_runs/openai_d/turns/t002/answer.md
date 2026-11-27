The architecture keeps drug inputs separate from measurement inputs, but both affect the recurrent state. That gives us possible places to edit; it does not establish any drug effect. Next I’ll inspect event timing, dose variation, and numeric encodings. Useful dose variation after accounting for prior readings would support estimating effects from the records.

```python
import numpy as np
import pandas as pd

train = pd.read_csv("/data/train.csv")
tuning = pd.read_csv("/data/tuning.csv")
train_ids = np.sort(train.stay_id.unique())
tuning_ids = np.sort(tuning.stay_id.unique())

print("=== DATA SIZE AND SCHEMA ===")
print("train:", train.shape, "stays:", len(train_ids))
print("tuning:", tuning.shape, "stays:", len(tuning_ids))
print(train.dtypes.to_string())
print("\nEvent counts:")
print(train.event.value_counts().to_string())

print("\n=== FIRST STAY, FIRST 35 ROWS ===")
print(train.loc[train.stay_id == train_ids[0]].head(35).to_string(index=False))
print("\nTime spacing / recording schedule:")
for event, g in train.groupby("event", sort=False):
    print(event, "hours:", np.sort(g.hour.unique())[:18],
          "rate range:", (g.rate.min(), g.rate.max()),
          "value range:", (g.value.min(), g.value.max()))

print("\n=== DRUG NUMERIC ENCODINGS ===")
for entry in VOCAB["numeric"]:
    if entry["code"] in ("insulin", "norepinephrine"):
        print(entry)

print("\n=== OTHER VOCAB STRUCTURE ===")
for key, val in VOCAB.items():
    if key not in ("config", "codes", "numeric", "event_kinds", "atom_kinds"):
        if isinstance(val, dict):
            print(key, "dict keys:", list(val.keys())[:20])
            if key == "value_bin_centres":
                for c in ("6", "7", "glucose_finger", "map_arterial"):
                    if c in val:
                        print(c, val[c])
        elif isinstance(val, list):
            print(key, "list length:", len(val), "first:", val[:2])
        else:
            print(key, val)

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in (
        "embed.numeric", "embed.bins", "encode_action", "attr_heads.value",
        "last_numeric", "z_post", "z_head", "embed_transition.value"
    )):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 2\n"
        "Architecture: actions have a separate DeepSets encoder, then enter "
        "the posterior and recurrent prelude. Reading heads also have "
        "same-code last-reading registers and transition logits. "
        "No drug effects inferred yet. Loaded train/tuning records and "
        "inspected schedule and drug encodings.\n"
    )
```