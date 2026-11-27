The architecture confirms where the dose wire, recurrent correction, and value readout must connect. The existing attention edit is still reconstruction-only. I’ll inspect the fitted controller objects and local simulation helpers next so I can test the complete connection without accidentally changing unrelated outputs.

```python
import inspect
import textwrap
import torch

g155 = dict(globals())

print("=== CONTROLLER OBJECTS ===")
for penalty, obj in controllers153.items():
    print("penalty", penalty)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if hasattr(value, "shape"):
                print(" ", key, tuple(value.shape))
            else:
                print(" ", key, repr(value)[:180])
    else:
        print(type(obj).__name__, repr(obj)[:300])

print("\n=== REMAINING RECENT VARIABLES ===")
for name, obj in sorted(g155.items()):
    if name > "flow" and name.endswith(("125", "145", "152", "153")):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict):
            print(name, "keys", list(obj)[:15])
        elif callable(obj):
            print(name, "callable")
        elif isinstance(obj, (int, float, str)):
            print(name, repr(obj)[:140])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))

print("\n=== CAUSAL / READOUT VARIABLES ===")
for name, obj in sorted(g155.items()):
    if any(s in name.lower() for s in ["kernel", "pole", "causal", "wire", "readout", "target"]):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(name, tuple(obj.shape),
                  repr(obj)[:300] if obj.numel() <= 24 and isinstance(obj, torch.Tensor)
                  else repr(obj)[:300] if isinstance(obj, np.ndarray) and obj.size <= 24
                  else "")
        elif isinstance(obj, dict):
            print(name, "keys", list(obj)[:15])
        elif isinstance(obj, (float, int, list, tuple)):
            print(name, repr(obj)[:350])

print("\n=== LOCAL HELPER SIGNATURES ===")
for name, obj in sorted(g155.items()):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            print(name, inspect.signature(obj))
        except Exception:
            pass

print("\n=== CAUSAL NOTE EXCERPTS ===")
lines155 = Path("/work/notes.md").read_text().splitlines()
hits155 = [i for i, line in enumerate(lines155)
           if any(s in line.lower() for s in
                  ["eight-lag", "discontinuity", "accepted immediate",
                   "provisional hour", "original rollout"])]
for i in hits155[-8:]:
    chunk = "\n".join(lines155[max(0, i-1):min(len(lines155), i+2)])
    print(textwrap.shorten(chunk, width=550, placeholder=" ..."))
```
