The records support an immediate glucose decrease from insulin and a MAP increase from norepinephrine. Their longer-term effects—especially insulin’s persistence—are less certain, so I won’t treat the estimated rollout targets as ground truth. I’ll locate the reconstruction functions without printing their large default arguments, then rebuild the dose sensitivities using the corrected register.

```python
import inspect
print("=== FUNCTION NAMES AND ARGUMENT NAMES ===")
for nm, ob in list(globals().items()):
    if inspect.isfunction(ob) and ob.__module__ == "__main__":
        try:
            print(nm, list(inspect.signature(ob).parameters))
        except Exception:
            print(nm)
print("\n=== CACHED ARRAYS AND DICTIONARIES ===")
for nm, ob in list(globals().items()):
    if nm.startswith("_") or nm in ["obj", "ob"]:
        continue
    if any(s in nm.lower() for s in ["head", "contrast", "prior", "cache", "state", "numeric", "cf", "hidden", "sample", "mc", "register"]):
        if hasattr(ob, "shape"):
            print(nm, type(ob).__name__, tuple(ob.shape))
        elif isinstance(ob, dict):
            print(nm, "keys:", list(ob.keys())[:12])
print("\n=== RECONSTRUCTION SOURCE ===")
for nm in ["reconstruct_reading_head", "score_view_local"]:
    ob = globals().get(nm)
    if inspect.isfunction(ob):
        print("\n", nm)
        try:
            print(inspect.getsource(ob)[:3300])
        except Exception as ex:
            print("Source unavailable:", type(ex).__name__, "file:", ob.__code__.co_filename)
print("\n=== RECENT ARRAY VARIABLE NAMES ===")
print([(nm, tuple(ob.shape)) for nm, ob in list(globals().items())
       if hasattr(ob, "shape") and not nm.startswith("_")][-45:])
```
