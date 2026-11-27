The architecture confirms that doses enter through a separate action encoder, while predictions also use decayed state, latent draws, and last-value registers. A mismatch in any of these could explain the reconstruction error. I’ll inspect the retained helpers and relevant tensors before choosing an edit; a correctly reconstructed head should reproduce the factual predictions without changing weights.

```python
import inspect, json
from pathlib import Path
import numpy as np

print("=== RELEVANT TENSORS ===")
prefixes = ("embed.", "encode_action.", "attr_heads.", "last_", "type_",
            "query_", "embed_transition.", "core.z_", "core.gap",
            "core.out_norm", "core.flow")
for name in NAMES:
    if name.startswith(prefixes) or any(s in name for s in ("prior", "post", "z_head", "last_numeric")):
        print(name, SHAPES[name])

print("\n=== RETAINED OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if inspect.isfunction(obj):
        print("FUNCTION", name, inspect.signature(obj),
              "uses", obj.__code__.co_names[:25])
    elif isinstance(obj, np.ndarray):
        print("ARRAY", name, obj.shape, str(obj.dtype))
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        print("TABLE", name, obj.shape, list(obj.columns)[:18])
    elif isinstance(obj, dict):
        print("DICT", name, list(obj)[:15])
    elif isinstance(obj, (list, tuple)) and len(obj) < 120:
        print("LIST", name, str(obj)[:300])

print("\n=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== COMPACT VOCAB ===")
print("keys:", list(VOCAB))
for key in ("config", "codes", "normalisation", "normalization", "categoricals"):
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key])[:2600])

print("\n=== HELPER SOURCE IF AVAILABLE ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and any(s in name.lower() for s in ("head", "predict", "recon", "last", "view")):
        try:
            print(name, inspect.getsource(obj)[:2000])
        except (OSError, TypeError):
            print(name, "source unavailable; constants:", str(obj.__code__.co_consts)[:700])
```
