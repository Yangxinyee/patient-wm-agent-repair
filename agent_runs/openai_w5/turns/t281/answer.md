The architecture confirms that doses can affect both recurrent state and future-reading feedback. Earlier record fits suggest nonzero effects, while the original model’s rollouts barely respond. I will now recover the existing channel and head parameters. This will let me test centering leakage without relying on the local likelihood reconstruction, which remains unvalidated.

```python
import inspect

r246_notes = Path("/work/notes.md").read_text()
print("NOTEBOOK CHARACTERS:", len(r246_notes))
print("=== NOTEBOOK SECTION OFFSETS ===")
r246_pos = 0
for r246_line in r246_notes.splitlines(keepends=True):
    if r246_line.startswith("## "):
        print(r246_pos, r246_line.strip())
    r246_pos += len(r246_line)

print("\n=== EDIT-RELATED PERSISTENT OBJECTS ===")
r246_items = []
for r246_name in sorted(list(globals())):
    if r246_name.startswith(("r245_", "r246_")):
        continue
    if not any(t in r246_name.lower() for t in
               ("center", "pair", "head", "gate", "coef", "causal", "target",
                "clean", "candidate", "exposure", "scale", "theta")):
        continue
    r246_obj = globals()[r246_name]
    if isinstance(r246_obj, (np.ndarray, torch.Tensor)):
        r246_desc = f"{type(r246_obj).__name__} {tuple(r246_obj.shape)}"
    elif isinstance(r246_obj, dict):
        r246_desc = "dict " + repr(list(r246_obj)[:12])
    elif callable(r246_obj):
        try:
            r246_desc = "function " + str(inspect.signature(r246_obj))
        except Exception:
            r246_desc = "callable"
    else:
        r246_desc = repr(r246_obj)[:140]
    r246_items.append(r246_name + ": " + r246_desc)
print("\n".join(r246_items)[:4200])

print("\n=== HEAD TENSOR SHAPES ===")
for r246_name in NAMES:
    if r246_name.startswith("attr_heads.value_bin"):
        print(r246_name, SHAPES[r246_name])

print("\n=== COMPACT EXISTING HEAD AUDIT ===")
for r246_sensor, r246_obj in globals().get("head_audit", {}).items():
    print(r246_sensor,
          {k: np.asarray(v).tolist() if np.asarray(v).size <= 8
           else {"shape": np.asarray(v).shape,
                 "mean": float(np.asarray(v).mean()),
                 "sd": float(np.asarray(v).std())}
           for k, v in r246_obj.items() if k != "ix"})
```
