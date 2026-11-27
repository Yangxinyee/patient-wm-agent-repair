The architecture confirms that doses affect recurrent state before the reading head; there is no direct dose-to-reading shortcut. The diagnostic output was truncated before showing the graft variables. I’ll inspect those now to distinguish a scaling error from a poor decoder—a scaling error should explain the unexpectedly large factual corrections.

```python
import inspect

snap = list(globals().items())
print("FUNCTIONS")
for name, obj in snap:
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)

print("\nARRAYS AND TABLES")
for name, obj in snap:
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if hasattr(obj, "shape"):
        print(name, type(obj).__name__, tuple(obj.shape))

print("\nRELEVANT DICTIONARIES / SCALARS")
for name, obj in snap:
    if any(s in name.lower() for s in
           ["graft", "gate", "decoder", "raw", "head", "probe", "ridge", "alpha", "beta", "scale"]):
        if isinstance(obj, dict):
            print(name, "keys", list(obj)[:18])
        elif isinstance(obj, (int, float, str, bool)):
            print(name, str(obj)[:200])
        elif isinstance(obj, (list, tuple)):
            print(name, str(obj)[:300])

print("\nRELEVANT TENSOR SHAPES")
for name in NAMES:
    if any(s in name for s in
           ["attr_heads.value", "last_numeric", "query_norm", "type_emb", "z_head", "z_prior"]):
        print(name, SHAPES[name])

notes_text = Path("/work/notes.md").read_text()
print("\nNOTEBOOK HEADINGS")
for line in notes_text.splitlines():
    if line.startswith("##"):
        print(line)
print("\nCURRENT EDITS", edited())
```