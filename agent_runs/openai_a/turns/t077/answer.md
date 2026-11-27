The inspection was truncated because a function’s default arguments included full weight tensors. That gives no new evidence about the graft. I’ll use bounded summaries to recover its feature and scaling variables, without changing weights.

```python
lines = []
snap = list(globals().items())

lines.append("FUNCTIONS (argument names only)")
for name, obj in snap:
    code_obj = getattr(obj, "__code__", None)
    if callable(obj) and getattr(obj, "__module__", None) == "__main__" and code_obj:
        lines.append(name + " " + str(code_obj.co_varnames[:code_obj.co_argcount]))

lines.append("\nMOST RECENT ARRAYS / TABLES")
arrs = [(name, obj) for name, obj in snap
        if not name.startswith("_") and hasattr(obj, "shape")]
for name, obj in arrs[-95:]:
    lines.append(f"{name}: {type(obj).__name__} {tuple(obj.shape)}")

lines.append("\nMOST RECENT DICTIONARIES")
dicts = [(name, obj) for name, obj in snap
         if isinstance(obj, dict) and name not in {"VOCAB", "SHAPES", "__builtins__"}]
for name, obj in dicts[-22:]:
    lines.append(f"{name}: {len(obj)} keys {list(obj)[:8]}")

lines.append("\nRECENT SMALL PARAMETERS")
for name, obj in snap[-160:]:
    if isinstance(obj, (float, int, bool, str)) and not name.startswith("_"):
        lines.append(f"{name}: {str(obj)[:100]}")

lines.append("\nHEAD TENSOR SHAPES")
for name in NAMES:
    if any(s in name for s in ["attr_heads.value", "last_numeric", "query_norm", "type_emb"]):
        lines.append(f"{name}: {SHAPES[name]}")
print("\n".join(lines)[:8500])
print("CURRENT EDITS", edited())
```