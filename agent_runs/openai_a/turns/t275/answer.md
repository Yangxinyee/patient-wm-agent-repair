The architecture confirms that the current edit changes reading predictions without changing the recorded treatment schedule. The inspection output was truncated, so I still need the cohort IDs, edit helpers, and causal-evidence summary. These will let me test the candidate against both the factual-fit limit and the effects supported by the records.

```python
import inspect
g = dict(globals())
print("=== FUNCTIONS ===")
print(" ".join(sorted(k for k,v in g.items()
                      if inspect.isfunction(v) and not k.startswith("_"))))
print("\n=== RELEVANT VARIABLES ===")
selected = [k for k in sorted(g) if any(s in k.lower() for s in
            ["checkpoint", "ids", "activity", "install", "source", "r3"])
            and not k.startswith("_") and not callable(g[k])]
for k in selected:
    v = g[k]
    if hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:8])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v), repr(v[:5])[:150])
    elif isinstance(v, str):
        print(k, "str length", len(v), repr(v[:120]))
    elif isinstance(v, (int,float,bool)):
        print(k, v)
print("\n=== EVIDENCE SUMMARY ===")
for block in notes_text.split("\n## "):
    if any(s in block.lower() for s in ["fuzzy", "discontinuity", "negative control", "norepinephrine identification"]):
        print(block[:1800])
        if len(block) > 1800:
            print("[section truncated]")
print("\n=== NOTEBOOK LENGTH ===", len(notes_text))
```
