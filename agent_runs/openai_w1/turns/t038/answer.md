The architecture provides a separate action encoder, but dose embeddings are shared across drugs. A broad edit could therefore affect both pathways and factual predictions. I’ll retrieve the normalization and retained analyses next; a defensible edit needs a measurable drug-specific response, not merely a correlation with dose.

```python
print("=== RETAINED OBJECTS ===")
r31_inventory = []
for r31_name, r31_obj in list(globals().items()):
    if r31_name.startswith(("_", "r30_", "r31_")):
        continue
    if callable(r31_obj):
        r31_desc = "callable"
    elif hasattr(r31_obj, "shape"):
        r31_desc = f"{type(r31_obj).__name__} {r31_obj.shape}"
    elif isinstance(r31_obj, dict):
        r31_desc = f"dict[{len(r31_obj)}] keys={list(r31_obj)[:10]}"
    elif isinstance(r31_obj, (list, tuple)):
        r31_desc = f"{type(r31_obj).__name__}[{len(r31_obj)}]"
    elif isinstance(r31_obj, (str, int, float, bool)):
        r31_desc = repr(r31_obj)[:90]
    else:
        continue
    r31_inventory.append(f"{r31_name}: {r31_desc}")
print("\n".join(r31_inventory)[:6500])

print("\n=== VOCAB STRUCTURE ===")
print("Top-level keys:", list(VOCAB))
for r31_key in VOCAB:
    if any(r31_s in r31_key.lower() for r31_s in
           ["code", "norm", "stat", "config", "unit"]):
        print(r31_key, repr(VOCAB[r31_key])[:2200])

print("\n=== ACTION / LATENT / REGISTER TENSORS ===")
for r31_name in NAMES:
    if (r31_name.startswith(("encode_action.", "embed.numeric",
                             "z_prior.", "z_post.", "z_head.",
                             "last_numeric.", "last_status."))
        or "filter" in r31_name):
        print(r31_name, SHAPES[r31_name])

print("\n=== NOTEBOOK BEGINNING ===")
print(Path("/work/notes.md").read_text()[:3200])
```
