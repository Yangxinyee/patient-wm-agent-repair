The architecture confirms two possible edit points: the action encoder and the reading head. Earlier estimates show that raw dose–reading correlations are heavily confounded, so I won’t use them as drug effects. I’ll recover the adjusted estimates and available helpers; this will determine which pathway is worth validating.

```python
start = notes_text.find("## Run 4")
end = notes_text.find("## Run 17", start)
print("=== CAUSAL FINDINGS ===")
print(notes_text[start:end if end > start else start + 5500][:5500])

print("\n=== HELPER NAMES ===")
helpers_now = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and getattr(obj, "__module__", None) == "__main__"
}
print(sorted(helpers_now))

print("\n=== RELEVANT ARRAYS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and any(
        s in name.lower() for s in ("wr_", "query", "gate", "head", "state")
    ):
        print(name, tuple(obj.shape))

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in (
        "type_emb", "query_norm", "attr_heads.value", "embed.code",
        "last_numeric", "last_status", "embed_transition", "encode_action", "z_post"
    )):
        print(name, SHAPES[name])
```
