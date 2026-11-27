The architecture confirms that drug doses reach measurement predictions through the recurrent state, not through a direct drug-to-reading register. That makes the action encoder and value head plausible edit locations. The output was truncated, so I’ll retrieve compact normalization and analysis details next; these should show whether dose information can be isolated without disturbing factual predictions.

```python
print("=== REMAINING RELEVANT TENSORS ===")
for name in NAMES:
    if name.startswith(("embed.numeric", "encode.", "encode_action.", "z_", "last_", "query_", "type_", "gap.", "filter_", "pending_", "embed_transition")):
        print(name, SHAPES[name])

print("\n=== VOCAB KEYS / CONFIG / CODES ===")
print(list(VOCAB.keys()))
for key in ("config", "codes", "normalization", "normalisation", "norm", "code_to_id"):
    if key in VOCAB:
        print(key, json.dumps(VOCAB[key])[:2800])
for key, val in VOCAB.items():
    if key not in {"config", "codes", "normalization", "normalisation", "norm", "code_to_id"}:
        if isinstance(val, dict):
            print(key, "keys:", list(val.keys())[:20])
        elif isinstance(val, list):
            print(key, "length:", len(val), "first:", repr(val[:2])[:450])

print("\n=== RETAINED DATAFRAMES ===")
for key, obj in list(globals().items()):
    if not key.startswith("_") and isinstance(obj, pd.DataFrame):
        print(key, obj.shape, list(obj.columns))
print("\n=== RETAINED ARRAYS, DICTS AND FUNCTIONS ===")
shown = 0
for key, obj in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(obj, np.ndarray) and shown < 30:
        print(key, obj.shape, str(obj.dtype)); shown += 1
    elif isinstance(obj, dict) and shown < 45:
        print(key, "keys:", list(obj.keys())[:12]); shown += 1
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(key, "function args:", obj.__code__.co_varnames[:obj.__code__.co_argcount])

print("\n=== FORWARD FILES ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])
print("\n=== FIXED TUNING SPLITS ===")
for key, obj in list(globals().items()):
    if not key.startswith("_") and isinstance(obj, (list, tuple, np.ndarray)) and any(x in key.lower() for x in ("stay", "dev", "check", "valid")):
        print(key, "length", len(obj), repr(obj[:8]))
```
