The architecture confirms that the edited value head changes predicted readings without changing the state-update machinery. The early regressions also show that untreated dose–outcome associations are badly confounded. I’ll recover the working caches and correction functions next; their contents should reveal whether the factual loss comes from the correction itself or a reconstruction error.

```python
print("=== RECENT CACHE/CANDIDATE VARIABLES ===")
for k in list(globals()):
    if k.startswith(("r199", "r200", "r201", "r202", "r203",
                     "c183", "c185", "c196", "r170", "r176", "r165")):
        v = globals()[k]
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", list(v)[:20])
        elif callable(v):
            print(k, "callable")
        else:
            print(k, type(v).__name__, str(v)[:100])
print("=== FUNCTIONS ===")
print([k for k in list(globals()) if callable(globals()[k])
       and getattr(globals()[k], "__module__", "") == "__main__"])
print("=== LAST 35 ARRAY VARIABLES ===")
arrnames = [k for k in list(globals()) if hasattr(globals()[k], "shape")]
for k in arrnames[-35:]:
    print(k, type(globals()[k]).__name__, tuple(globals()[k].shape))
print("=== CAUSAL EVIDENCE, RUNS 4–8 ===")
a = _notes.find("## Run 4")
b = _notes.find("## Run 9", a)
print(_notes[a:min(b if b > a else a+3300, a+3300)])
print("=== CURRENT EDITS ===", edited())
```
