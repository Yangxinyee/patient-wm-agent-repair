The architecture confirms that the current edits affect only the reading-value head, leaving recurrent states unchanged. That should allow factual-loss comparisons on cached raw head inputs. The recovery output was truncated, so I’ll inspect the live variables first. I need both the raw probe coordinates and the saved candidate weights before testing a corrected direction.

```python
print("=== RECENT LIVE VARIABLES ===")
for k in sorted(list(globals())):
    if k.startswith(("c18", "c19", "c20", "c21", "c22", "r18", "r19", "r20", "r21", "r22")):
        v = globals()[k]
        print(k, type(v).__name__, getattr(v, "shape", ""),
              list(v.keys())[:12] if isinstance(v, dict) else "")
print("=== FUNCTIONS ===")
print(" ".join(sorted(k for k, v in list(globals().items())
                      if callable(v) and not isinstance(v, type)
                      and not k.startswith("__"))))
print("=== OTHER RELEVANT VARIABLE NAMES ===")
print(" ".join(sorted(k for k in globals()
                     if any(s in k.lower() for s in
                            ["probe", "candidate", "cache", "raw", "direction", "r124", "r147", "transform"])
                     and not k.startswith("__"))))
print("=== CAUSAL/INITIAL EDIT NOTES ===")
print(Path("/work/notes.md").read_text()[4700:10000])
```
