Using seconds for memory ages restores agreement: summary RMSE is about 0.000014, with only a small recurrent-state rounding error. The recurrent replay is now usable. The readout helpers contain mismatched batch sizes, so I’ll rebuild a consistent reading batch and check it against factual predictions before calibrating a correction.

```python
print("=== FACTUAL OUTPUT ===")
print(F97.head(8).to_string(index=False))
print("columns", list(F97.columns))
print("\n=== READOUT HELPERS ===")
print(compact_bytecode(means104))
print("\n=== READOUT DATA INVENTORY ===")
for name in sorted(list(globals())):
    if name.endswith(("97", "104", "105", "110")):
        obj = globals()[name]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            arr = as_numpy(obj)
            print(name, arr.shape, "first", arr.reshape(-1)[:4])
        elif isinstance(obj, dict):
            print(name, "dict", list(obj)[:8])
        elif isinstance(obj, str):
            print(name, "str", len(obj), obj[:50].replace("\n"," "))
print("\n=== VOCAB REGISTERS ===")
print("config", VOCAB.get("config"))
print("codes", VOCAB.get("codes"))
print("categorical vocab", VOCAB.get("categorical", "absent"))
print("\n=== REGISTRY WEIGHTS ===")
for name in NAMES:
    if name.startswith(("last_", "type_emb", "embed_transition")):
        print(name, SHAPES[name])
print("\n=== NOTEBOOK READOUT MENTIONS ===")
for line in notes.splitlines():
    if any(s in line.lower() for s in
           ("readout", "register", "last_numeric", "factual mean", "value head")):
        print(line[:650])
```
