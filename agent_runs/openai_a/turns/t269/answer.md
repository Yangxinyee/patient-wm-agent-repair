No weights changed. The diagnostic was overwhelmed by tensor-valued function defaults, so it did not expose the gate recipe. I will use a bounded inventory next; finding the saved separator and exact correction construction will let me test the activity gate without risking the validated fallback.

```python
out_lines = ["=== FUNCTIONS: ARGUMENT NAMES ONLY ==="]
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        c = obj.__code__
        out_lines.append(f"{name}({', '.join(c.co_varnames[:c.co_argcount])})")

out_lines.append("\n=== RELEVANT NON-FUNCTION VARIABLES ===")
for name, obj in sorted(list(globals().items())):
    if not re.search(r"active|activity|gate|sep|r3|checkpoint|cache|panel|finger", name, re.I):
        continue
    if name in {"obj", "out_lines"} or inspect.isfunction(obj):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        desc = f"{type(obj).__name__} {tuple(obj.shape)}"
    elif isinstance(obj, dict):
        desc = f"dict {len(obj)} keys={repr(list(obj.keys())[:8])[:180]}"
    elif hasattr(obj, "columns") and hasattr(obj, "shape"):
        desc = f"table {obj.shape} columns={list(obj.columns)[:8]}"
    elif isinstance(obj, (str, int, float, bool, tuple)):
        desc = repr(obj)[:160]
    else:
        desc = type(obj).__name__
    out_lines.append(f"{name}: {desc}")
print("\n".join(out_lines)[:8000])

print("\n=== RECENT IDENTIFICATION NOTES ===")
hit_lines = []
for i, s in enumerate(lines):
    if re.search(r"fuzzy.RD|persistence|NE.*(null|effect|estimate)|norepinephrine.*(null|effect)", s, re.I):
        hit_lines.append(s)
print("\n".join(hit_lines[-16:])[:1700])
```
