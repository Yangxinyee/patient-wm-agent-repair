The architecture confirms that value-head edits leave the other prediction heads unchanged directly, although generated readings can still affect later states. The earlier records also show that simple dose associations are misleading. I’ll recover the identification evidence and exact head-edit construction before designing another candidate; a useful correction must generalize across stays, not merely fit the four cached interventions.

```python
import re
import inspect

sections232 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
for needle in ["Causal identification checks", "Sensor-proxy dynamic check",
               "Fixed-dose response-kernel estimates",
               "Run 159:", "Run 177:", "Run 192:", "Run 224:"]:
    matches = [s for s in sections232 if needle in s.splitlines()[0]]
    for s in matches:
        print(s[:1400])

print("\nCACHE DIMENSIONS")
for name in ["wm_train145", "wm_tune145", "wm_cf146", "map192_candidates",
             "map224_candidates", "map218_candidates"]:
    obj = globals().get(name)
    if isinstance(obj, dict):
        print(name, [(str(k), tuple(v.shape) if hasattr(v, "shape")
                      else type(v).__name__) for k, v in list(obj.items())[:14]])
    else:
        print(name, type(obj).__name__)

print("\nRELEVANT FUNCTIONS")
for name in sorted(list(globals())):
    obj = globals()[name]
    if callable(obj) and name.startswith(("head177", "head224", "head225",
                                         "mc_", "wm_", "map192", "map224",
                                         "scalar_eval", "response_design")):
        if hasattr(obj, "__code__"):
            c = obj.__code__
            print(name, inspect.signature(obj), "uses:", c.co_names[:25])

print("\nEDITED TENSOR SHAPES")
for name in edited():
    print(name, tuple(load(name).shape))
print("\nHEAD CONSTRUCTION CONSTANTS")
for name in ["head177_make", "scalar_eval208", "head224_eval"]:
    obj = globals().get(name)
    if callable(obj) and hasattr(obj, "__code__"):
        print(name, "constants:", repr(obj.__code__.co_consts)[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 232: selective-correction audit\n"
            "- Architecture re-read: restricting edits to the value head avoids "
            "direct changes to other heads, but rollout feedback remains relevant.\n"
            "- Recovering causal identification assumptions and exact head construction; "
            "no weights changed.\n")
```