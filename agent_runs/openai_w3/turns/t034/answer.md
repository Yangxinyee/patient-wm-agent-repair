The architecture confirms that doses affect the model through action embeddings and recurrent state updates, not through a direct drug-effect coefficient. The output was cut off before the live helpers, so I still lack enough detail to compute a safe edit. I’ll inspect only that missing metadata next; it should either reveal the reconstruction error or give us a route to a narrowly targeted edit.

```python
print("=== REMAINING TENSOR FAMILIES ===")
for name in NAMES:
    if name.startswith(("encode", "z_", "gap", "last_", "query_", "type_", "embed.numeric", "embed_transition", "monitor", "pending", "filter")):
        print(name, tuple(SHAPES[name]))
print("=== VOCAB SUMMARY ===")
for key, value in VOCAB.items():
    if key in ("value_bin_edges", "value_bin_centres", "dose_bin_edges", "rate_bin_edges",
               "dose_bin_centres", "rate_bin_centres"):
        continue
    text = repr(value)
    print(key, text[:1800])
print("=== LIVE NAMES ===")
parts = []
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if callable(value):
        parts.append(key + "()")
    elif hasattr(value, "shape"):
        parts.append(key + ":" + str(tuple(value.shape)))
    elif isinstance(value, dict):
        parts.append(key + ":dict(" + str(len(value)) + ")")
    elif isinstance(value, (list, tuple)):
        parts.append(key + ":" + type(value).__name__ + "(" + str(len(value)) + ")")
print("\n".join(parts))
with open("/work/notes.md", "a") as f:
    f.write("\nPython run 29; forwards 3. Architecture confirms action-embedding and posterior/state pathways. Recovering only missing metadata to avoid output truncation. No edits.\n")
```