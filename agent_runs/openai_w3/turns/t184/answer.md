The candidate changes are concentrated in the two intended latent coordinates, but that does not yet establish a drug effect. The diagnostic stopped without changing weights. I’ll recover the carrier and replay definitions before testing the complete transport path. A dose-dependent response with little factual disturbance would support this approach; substantial factual distortion would refute it.

```python
import os, inspect, re
print("=== ARCHITECTURE ===")
print(open("/model/ARCHITECTURE.md").read())
print("\n=== CURRENT EDITS ===", edited())

print("\n=== CANDIDATE AND REPLAY INVENTORY ===")
for key in sorted(list(globals())):
    if re.search(r"(154|155|156|157|158|159|160|161|162|163)$", key):
        obj = globals()[key]
        if callable(obj):
            try:
                print(key, inspect.signature(obj))
            except Exception:
                print(key, "callable")
        elif isinstance(obj, dict):
            print(key, "dict", [(k, tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                                for k, v in list(obj.items())[:12]])
        elif hasattr(obj, "shape"):
            print(key, tuple(obj.shape))
        elif isinstance(obj, (str, int, float, bool, list, tuple)):
            print(key, repr(obj)[:500])

print("\n=== IDENTIFICATION / TRANSPORT NOTES ===")
notes165 = open("/work/notes.md").read()
for paragraph165 in notes165.split("\n\n"):
    if re.search(r"(identif|random|instrument|kernel|confound|effect|zero-payload|Python run 154)", paragraph165, re.I):
        print(paragraph165[:2200])
print("\n=== RECENT NOTES ===")
print(notes165[-3500:])
```
