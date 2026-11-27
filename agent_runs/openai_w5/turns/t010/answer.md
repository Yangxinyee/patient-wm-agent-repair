The architecture offers two possible edit points: the drug-action pathway and the value head. The value head also relies on the last reading, which may explain why factual accuracy does not guarantee useful dose responses. I’ll inspect the remaining tensors, normalisation tables, and saved analysis variables before choosing an edit. We need a dose-sensitive signal that can be changed without broadly disturbing other predictions.

```python
print("=== REMAINING / HEAD TENSORS ===")
for name in NAMES:
    if name.startswith(("heads", "last", "pending", "query", "type", "z_", "filter", "monitor", "time", "exec", "embed.numeric", "embed_transition", "encode", "encode_action")):
        print(name, SHAPES[name])
print("\n=== VOCAB ===")
for k,v in VOCAB.items():
    if k == "config" or "code" in k or "norm" in k or "centre" in k or "edge" in k:
        print(k, json.dumps(v)[:4200])
    else:
        print(k, type(v).__name__, str(v)[:180])
print("\n=== KERNEL INVENTORY ===")
inventory = []
for k,v in list(globals().items()):
    if not k.startswith("_") and k not in {"NAMES","SHAPES","VOCAB","inventory"}:
        if isinstance(v, pd.DataFrame):
            print(k, "DataFrame", v.shape, list(v.columns))
        elif isinstance(v, dict):
            print(k, "dict", list(v.keys())[:18])
        elif isinstance(v, np.ndarray) and v.size > 100:
            print(k, "array", v.shape, str(v.dtype))
        elif isinstance(v, (list,tuple)) and len(v) >= 10:
            print(k, type(v).__name__, len(v), str(v[:5])[:180])
        else:
            inventory.append((k,type(v).__name__))
print("other variables:", inventory)
print("\n=== PANEL SAMPLES ===")
print(panel.head(4).to_string())
print("\n=== FORWARD REQUEST METADATA ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])
```
