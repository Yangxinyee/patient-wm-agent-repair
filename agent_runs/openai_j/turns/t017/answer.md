The architecture confirms that action embeddings affect shared state and therefore many predictions. Editing only the value head could be safer, provided its features retain usable dose information. The output was truncated, so I’ll inspect the available states, normalization tables, and retained analysis objects before choosing an edit.

```python
import pandas as pd
import json
from pathlib import Path

print("=== RELEVANT TENSORS ===")
prefixes = ("embed.numeric", "encode_action", "last_", "z_", "query_", "type_emb",
            "embed_transition", "encode.")
for name in NAMES:
    if name.startswith(prefixes):
        print(name, SHAPES[name])

print("\n=== VOCAB ===")
print(json.dumps(VOCAB, indent=1)[:11500])

print("\n=== RETAINED DATA / ARRAYS / DICTS ===")
items_snapshot = list(globals().items())
for key, obj in items_snapshot:
    if key.startswith("_") or key in {"VOCAB", "SHAPES", "items_snapshot"}:
        continue
    if isinstance(obj, pd.DataFrame):
        print(key, "DataFrame", obj.shape, "columns:", list(obj.columns)[:14])
    elif isinstance(obj, np.ndarray):
        print(key, "array", obj.shape)
    elif isinstance(obj, dict) and key not in {"__builtins__"}:
        print(key, "dict keys:", list(obj)[:12])

print("\n=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    for path in folder.iterdir():
        print(str(path))
        if path.suffix == ".npz":
            with np.load(path) as archive:
                print({k: archive[k].shape for k in archive.files})
                for k in archive.files:
                    if archive[k].ndim == 1:
                        print(k, archive[k][:5])
        elif path.suffix == ".csv":
            frame = pd.read_csv(path)
            print(frame.shape, list(frame.columns))
            print(frame.head(2).to_string(index=False))
```
