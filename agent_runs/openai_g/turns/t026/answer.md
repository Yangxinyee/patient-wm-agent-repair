The repeated output contains no new validation result. I’ll make the recovery output shorter so we can see what the current edits actually did. Near-zero rollouts with an NLL increase below 0.005 would support keeping this targeted approach.

```python
import inspect
_recovery_text = open("/work/notes.md").read()
print("NOTEBOOK LENGTH:", len(_recovery_text))
print("=== PRIOR EDIT AND VALIDATION NOTES ===")
for _p in _recovery_text.split("\n\n"):
    if any(_s in _p.lower() for _s in ["forward 2", "forward 3", "forward 4", "flatten", "erasure", "nll", "projection"]):
        print(_p[:1800])
print("=== FUNCTIONS ===")
print([(k, str(inspect.signature(v))) for k, v in list(globals().items())
       if not k.startswith("_") and inspect.isfunction(v)])
print("=== ARRAY / DATAFRAME VARIABLES ===")
print([(k, type(v).__name__, tuple(v.shape)) for k, v in list(globals().items())
       if not k.startswith("_") and hasattr(v, "shape")])
print("=== DICTIONARY VARIABLES ===")
print([(k, len(v), list(v)[:5]) for k, v in list(globals().items())
       if not k.startswith("_") and isinstance(v, dict) and k not in ["SHAPES", "VOCAB"]])
print("=== FORWARD FILES ===")
for _d in sorted(os.listdir("/work")):
    if _d.startswith("fwd_"):
        print(_d, os.listdir("/work/" + _d))
```