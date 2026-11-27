The architecture confirms that drugs enter through a separate action encoder, while reading predictions also depend strongly on prior readings. This gives us places to correct dose sensitivity without replacing the whole model. The previous output was truncated, so I’ll retrieve only the cached functions and tensors needed for that correction.

```python
print("=== LOCAL FUNCTIONS ===")
for name in sorted(list(globals())):
    obj = globals()[name]
    if not name.startswith("_") and callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name)

print("\n=== STATE / HEAD / ALIGNMENT CACHES ===")
for name in sorted(list(globals())):
    if name.startswith("_"):
        continue
    obj = globals()[name]
    if any(k in name.lower() for k in
           ["state", "head", "feat", "prior", "align", "config", "factual", "target"]):
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj)[:12])
        elif isinstance(obj, (list, tuple)):
            print(name, "len", len(obj), repr(obj)[:180])

print("\n=== EDITABLE PATHWAYS ===")
for name in NAMES:
    if any(name.startswith(p) for p in
           ["embed.numeric", "embed.bins", "encode_action", "last_numeric",
            "attr_heads.value", "embed_transition.value", "z_", "query_norm",
            "type_emb", "last_status", "gap", "core.flow"]):
        print(name, SHAPES[name])

print("\n=== FORWARD FILES ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
print("Current edits:", edited())
```
