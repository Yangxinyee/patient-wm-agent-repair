The architecture exposes rate and dose through numeric and bin embeddings. We may be able to remove dose sensitivity while retaining drug identity and presence.

The previous output was truncated, so I’ll inspect just the relevant tensors, normalization tables, and surviving variables. The proposed edit should give identical rollouts across dose multipliers; a factual NLL increase above 0.005 would reject it.

```python
import inspect
import json
import numpy as np

print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if (
        name.startswith(("embed.", "encode_action."))
        or any(s in name for s in ("last_numeric", "last_status", "value_bin"))
    ):
        print(name, SHAPES[name])

print("\n=== VOCAB TOP-LEVEL STRUCTURE ===")
for key, value in VOCAB.items():
    if isinstance(value, dict):
        print(key, "dict keys:", list(value.keys()))
    elif isinstance(value, list):
        print(key, "list length:", len(value))
    else:
        print(key, str(value)[:200])

for key in ("config", "codes"):
    if key in VOCAB:
        print("\n", key, json.dumps(VOCAB[key], default=str))

print("\n=== NORMALIZATION AND UNIT TABLES ===")
for key, value in VOCAB.items():
    if any(s in key.lower() for s in ("norm", "unit", "categor")):
        print(key, json.dumps(value, default=str)[:2200])

print("\n=== SURVIVING ARRAYS / TABLES ===")
for name, obj in list(globals().items()):
    if not name.startswith("_") and hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape, getattr(obj, "dtype", ""))

print("\n=== SURVIVING USER FUNCTIONS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            print(name, inspect.signature(obj))
        except (ValueError, TypeError):
            print(name)

print("\n=== NUMERIC EMBEDDING COLUMN NORMS ===")
for name in NAMES:
    if name.startswith("embed.numeric"):
        tensor = np.asarray(load_original(name))
        print(name, tensor.shape)
        if tensor.ndim == 2:
            print("column norms:", np.linalg.norm(tensor, axis=0))
        else:
            print("norm:", np.linalg.norm(tensor))

print("\nCurrent edits:", edited())
```