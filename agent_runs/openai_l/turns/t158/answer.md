The last run found usable private-state write routes, but their background activity could harm factual predictions. The attention reconstruction is factual-fit neutral; it has not yet repaired dose responses. I’ll recover the record-supported effects and controller interfaces before assembling a correction. Success requires both the right dose response and factual NLL within the allowed tolerance.

```python
import os, json, inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())

print("\n=== NOTEBOOK: EARLY EVIDENCE ===")
notes146 = open("/work/notes.md").read()
print(notes146[:16000])
print("\nNOTEBOOK LENGTH:", len(notes146))
print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes146.splitlines() if line.startswith("#")))

print("\n=== AVAILABLE FUNCTIONS ===")
for name146, obj146 in sorted(list(globals().items())):
    if inspect.isfunction(obj146) and not name146.startswith("_"):
        try:
            sig146 = str(inspect.signature(obj146))
        except Exception:
            sig146 = "(signature unavailable)"
        print(name146, sig146)

print("\n=== CONTROLLER / CAUSAL VARIABLES ===")
for name146, obj146 in sorted(list(globals().items())):
    if name146.endswith(("125", "145")) or any(
        term146 in name146.lower()
        for term146 in ("kernel", "causal", "target", "route", "private", "effect")
    ):
        if isinstance(obj146, np.ndarray):
            print(name146, "array", obj146.shape, obj146.dtype,
                  "norm", round(float(np.linalg.norm(obj146)), 5))
            if obj146.size <= 80:
                print(obj146)
        elif isinstance(obj146, dict):
            print(name146, "dict keys", list(obj146.keys())[:30])
        elif isinstance(obj146, (list, tuple)):
            print(name146, type(obj146).__name__, "length", len(obj146),
                  "sample", repr(obj146[:2])[:400])
        elif not inspect.isfunction(obj146):
            print(name146, type(obj146).__name__, repr(obj146)[:250])
```
