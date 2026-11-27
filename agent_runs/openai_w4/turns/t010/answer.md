The architecture shows that doses enter the recurrent state, while measurement predictions also use a last-reading register. That suggests a targeted readout edit might work—but only if the state retains usable dose information. The output was truncated, so I’ll inspect the remaining tensors and analysis objects compactly before testing that hypothesis.

```python
import json
import numpy as np
import pandas as pd

print("=== REMAINING TENSORS ===")
start = NAMES.index("embed.numeric.bias") if "embed.numeric.bias" in NAMES else 110
for name in NAMES[start:]:
    print(name, SHAPES[name])

print("\n=== VOCAB SUMMARY ===")
print("Top-level keys:", list(VOCAB))
for key in ("config", "codes"):
    print(key, json.dumps(VOCAB.get(key), default=str))
for key, val in VOCAB.items():
    if key in ("config", "codes"):
        continue
    if isinstance(val, dict):
        items = list(val.items())
        print(key, "keys:", list(val)[:15], "sample:", str(items[:2])[:650])
    else:
        print(key, str(val)[:650])

print("\n=== RETAINED ANALYSIS OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in ("obj", "val", "VOCAB", "SHAPES", "NAMES"):
        continue
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, "columns:", list(obj.columns)[:28])
    elif isinstance(obj, np.ndarray) and obj.size > 100:
        print(name, obj.shape, obj.dtype)
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        print(name, "function")
print("Names:", sorted(k for k in globals() if not k.startswith("_")))

print("\nUnique tuning IDs:",
      len(tuning_ids_repair), tuning_ids_repair[:5], tuning_ids_repair[-5:])
train_id_preview = pd.read_csv("/data/train.csv", usecols=["stay_id"], nrows=10000)
print("Training ID preview:", train_id_preview.stay_id.unique()[:10].tolist())
print("Edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 7\n"
        "Architecture inspection: drugs enter posterior and recurrent prelude; "
        "value heads also have a same-code last-reading register.\n"
        "Considering targeted readout repair, conditional on states retaining dose information.\n"
        "Run 6 output was truncated; obtained compact remaining tensor/vocab/object summaries.\n"
        "No weights edited.\n"
    )
```