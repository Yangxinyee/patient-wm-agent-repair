The records support a negative insulin response and a positive norepinephrine response, but insulin’s longer-term magnitude remains uncertain. Simple dose-embedding gains did not recover those responses. I’ll inspect the posterior pathway for a history-conditioned dose signal; it is promising only if it is small on factual records and responsive to changed doses.

```python
print("=== REMAINING RECONSTRUCTION NOTES ===")
print(notes_text[9500:12500])

print("\n=== FUNCTION SIGNATURES ===")
funcs = []
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and getattr(obj, "__module__", "") == "__main__" and hasattr(obj, "__code__"):
        funcs.append((name, obj.__code__.co_varnames[:obj.__code__.co_argcount]))
print(funcs)

print("\n=== POSTERIOR, ENCODER AND VALUE-HEAD TENSORS ===")
for name in NAMES:
    if any(s in name for s in
           ["z_post", "z_prior", "z_in", "z_head", "encode_action",
            "embed.categorical", "attr_heads.value", "last_numeric"]):
        print(name, SHAPES[name])

print("\n=== SELECTED OBJECT STRUCTURES ===")
for name in ["head_meta", "head_factual", "head_hidden", "feat0", "hm",
             "end_records", "bundles", "bundle_records", "reconstruction",
             "train", "tuning", "panel", "W", "weights", "cache"]:
    if name not in globals():
        continue
    obj = globals()[name]
    print(name, type(obj).__name__, getattr(obj, "shape", None))
    if isinstance(obj, pd.DataFrame):
        print("columns:", obj.columns.tolist(), "first row:", obj.head(1).to_dict("records"))
    elif isinstance(obj, dict):
        print("keys:", list(obj)[:12])
print("Active edits:", edited())
```
