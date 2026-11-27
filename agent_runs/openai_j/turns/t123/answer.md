The sampled reconstruction now closely matches the original GPU predictions. That resolves the earlier factual-fit calculation error, but the cached dose contrasts may still use the wrong inputs. I’ll recover the reconstruction helpers and causal estimates, then rebuild those contrasts. Agreement with the GPU on unchanged records is the required safety check.

```python
import inspect, os, json
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== EARLY NOTEBOOK: RECORD EVIDENCE ===")
print(open("/work/notes.md").read()[:15000])
print("\n=== RESULT TABLES ===")
for fn in ["dynamic_iv.csv", "sensor_iv.csv", "kernel_targets.csv", "local_projections.csv"]:
    p = "/work/" + fn
    if os.path.exists(p):
        t = pd.read_csv(p)
        print(fn, t.shape)
        print(t.head(18).to_string(index=False))
print("\n=== RECONSTRUCTION HELPERS ===")
for name, obj in list(globals().items()):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        if any(s in name.lower() for s in ["head", "state", "contrast", "numeric", "readout", "posterior", "prior", "reconstruct", "action", "local"]):
            print("\nFUNCTION", name, inspect.signature(obj))
            try:
                print(inspect.getsource(obj)[:10000])
            except Exception:
                pass
print("\n=== RELEVANT CACHED OBJECTS ===")
for name, obj in list(globals().items()):
    if any(s in name.lower() for s in ["head", "contrast", "prior", "numeric", "mc", "cache", "state"]):
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj.keys())[:20])
print("Active edits:", edited())
```
