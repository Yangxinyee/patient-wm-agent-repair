The architecture confirms that dose information reaches a shared state, whereas the value head also uses each sensor’s last-reading register. The notebook reports effects supported by dosing-threshold comparisons, but those estimates need to stay separate from the rejected state-space estimates. I’ll recover the accepted effect kernel and live fitting objects before changing weights.

```python
import re, json, inspect
text230 = Path("/work/notes.md").read_text()
start230 = text230.find("## Run 9")
end230 = text230.find("## Run 22")
print("=== RECORD IDENTIFICATION ===")
print(text230[start230:end230][:5200])

funcs230 = []
objs230 = []
for k in sorted(list(globals())):
    if k.startswith("_"):
        continue
    v = globals()[k]
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            funcs230.append(k + str(inspect.signature(v)))
        except Exception:
            funcs230.append(k)
    if re.search(r"(219|22[2-8]|208|212|200|179|kernel|canonical|backup|direction|quiet|audit)", k, re.I):
        if isinstance(v, np.ndarray):
            objs230.append(f"{k}: array {v.shape} {v.dtype}")
        elif isinstance(v, dict):
            objs230.append(f"{k}: dict {list(v.keys())[:12]}")
        elif hasattr(v, "shape"):
            objs230.append(f"{k}: {type(v).__name__} {tuple(v.shape)}")
        else:
            objs230.append(f"{k}: {type(v).__name__} {str(v)[:65]}")
print("\n=== FUNCTIONS ===")
print("\n".join(funcs230))
print("\n=== RECENT OBJECTS ===")
print("\n".join(objs230))
```
