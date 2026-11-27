```python
import os, inspect, numpy as np, torch
print(open('/model/ARCHITECTURE.md').read())
print("\nRECENT VARIABLES")
for k in sorted(list(globals())):
    if any(s in k.lower() for s in ['136', '135', '134', '133', 'pilot', 'inject', 'response', 'bdelta', 'notes']):
        v = globals()[k]
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(k, type(v).__name__, tuple(v.shape))
        elif callable(v):
            try: print(k, inspect.signature(v))
            except Exception: print(k, "callable")
        elif isinstance(v, (dict, list, tuple)):
            print(k, type(v).__name__, len(v), str(v)[:350])
        else:
            print(k, type(v).__name__, str(v)[:350])
print("\nRECENT NOTEBOOK CONTEXT")
notes_recovery = open('/work/notes.md').read()
print(notes_recovery[:16000])
print("\nCURRENT EDITS", edited())
```
