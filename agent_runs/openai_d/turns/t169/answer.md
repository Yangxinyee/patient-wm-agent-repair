The records support opposite-to-naive adjusted effects: insulin lowers glucose, while norepinephrine raises MAP. The strength-2 insulin edit passed one factual check, but the combined trial did not. I’ll inspect the cached inputs and correction formulas to identify which component is causing the held-out loss.

```python
print("=== RECENT DIAGNOSTIC OBJECTS ===")
for _n in sorted(list(globals())):
    if re.match(r"r(13[6-9]|14[0-9])", _n):
        _o = globals()[_n]
        if hasattr(_o, "shape"):
            print(_n, type(_o).__name__, tuple(_o.shape))
        elif isinstance(_o, dict):
            print(_n, "dict", list(_o.keys())[:12])
        elif isinstance(_o, (list, tuple)):
            print(_n, type(_o).__name__, len(_o))
        elif callable(_o):
            print(_n, "callable")
        else:
            print(_n, repr(_o)[:180])

print("=== FUNCTIONS ===")
print([n for n, o in list(globals().items())
       if callable(o) and not n.startswith("_") and getattr(o, "__module__", "") == "__main__"])

print("=== OTHER ARRAY / TABLE OBJECTS ===")
for _n in sorted(list(globals())):
    if _n.startswith("_") or re.match(r"r\d+", _n):
        continue
    _o = globals()[_n]
    if hasattr(_o, "shape"):
        print(_n, type(_o).__name__, tuple(_o.shape))
print("=== CANDIDATE KEYS ===")
for _n in ["r107_backup", "r115_strength2", "r139_fallback", "r139_candidate",
           "r145_original", "r145_current"]:
    if _n in globals():
        _o = globals()[_n]
        print(_n, list(_o.keys()) if isinstance(_o, dict) else type(_o).__name__)
```
