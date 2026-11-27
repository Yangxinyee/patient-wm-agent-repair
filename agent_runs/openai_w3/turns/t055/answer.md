The architecture offers several possible edit points, but a posterior-only correction would still pass through nonlinear state updates. Before changing weights, I’ll inspect the retained dose decoders, response directions and original rollout results. A reliable decoder and stable response direction would support a small pilot; otherwise I’ll choose a different mechanism.

```python
import inspect
import numpy as np

print("=== HELPERS ===")
for k, v in list(globals().items()):
    if inspect.isfunction(v) and not k.startswith("_"):
        try:
            print(k, str(inspect.signature(v))[:180])
        except Exception:
            pass

print("\n=== RECENT VARIABLES ===")
items = [(k, v) for k, v in list(globals().items())
         if not k.startswith("_") and k not in ("In", "Out")]
for k, v in items[-105:]:
    if isinstance(v, dict):
        print(k, "dict", len(v), [str(z)[:50] for z in list(v)[:6]])
    elif hasattr(v, "shape") and not callable(v):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, (str, int, float, list, tuple)):
        print(k, str(v)[:110])

print("\n=== ORIGINAL ROLLOUT ===")
print(roll0.head(3).to_string(index=False))
print("columns:", list(roll0.columns))
print("\n=== POSTERIOR AND VALUE HEAD SHAPES ===")
for n in NAMES:
    if any(s in n for s in ("z_post", "z_in", "z_head", "attr_heads.value_bin", "encode_action")):
        print(n, SHAPES[n])

print("\n=== EARLY NOTEBOOK ===")
print(open("/work/notes.md").read()[:2200])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 50; forwards 3. Architecture reread. Recovering compact "
            "helper/fit metadata and original dose-rollout scale; no edits.\n")
```