The architecture confirms that doses affect the shared recurrent state, so a broad edit could disturb more than glucose or MAP. The output was truncated before the causal checks and reconstruction helpers appeared. I’ll retrieve those sections separately; I need both a defensible effect target and a faithful reconstruction before testing another edit.

```python
import inspect
notes_text = open("/work/notes.md").read()
print("=== CAUSAL CHECKS ===")
start = notes_text.find("## Causal identification checks")
print(notes_text[start:start + 4500])
print("\n=== FITTED EFFECT TARGETS ===")
for fn in ["kernel_targets.csv", "sensor_iv.csv", "dynamic_iv.csv"]:
    tab = pd.read_csv("/work/" + fn)
    print(fn, tab.shape)
    print(tab.head(7).to_string(index=False))
print("\n=== LOCAL FUNCTION SIGNATURES ===")
for nm, ob in list(globals().items()):
    if inspect.isfunction(ob) and ob.__module__ == "__main__":
        try:
            print(nm, inspect.signature(ob))
        except Exception:
            print(nm)
print("\n=== HEAD/STATE CACHE NAMES ===")
for nm, ob in list(globals().items()):
    if any(s in nm.lower() for s in ["head", "contrast", "prior", "cache", "state", "numeric", "cf_"]):
        if hasattr(ob, "shape"):
            print(nm, type(ob).__name__, tuple(ob.shape))
        elif isinstance(ob, dict):
            print(nm, "keys:", list(ob.keys())[:12])
print("Active edits:", edited())
```
