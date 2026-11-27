The helpers support an edit in the action-token attention path, which has appreciable influence on the state. That is more promising than the nearly ignored latent pathway. I’ll inspect the retained dose-contrast fit and calibration code; a useful fit should track dose changes while producing little correction on recorded doses.

```python
import inspect, dis, io
from pathlib import Path

print("=== EDIT AND CALIBRATION HELPERS ===")
for nm131 in ("r66_weight", "prelude_cache88", "parts105", "means105", "next_view105"):
    fn131 = globals()[nm131]
    try:
        src131 = inspect.getsource(fn131)
    except (OSError, TypeError):
        buf131 = io.StringIO()
        dis.dis(fn131, file=buf131)
        src131 = buf131.getvalue()
    print("\n---", nm131, "---")
    print(src131[:2200])

print("\n=== KEY ARRAYS ===")
for nm131 in ("h81", "e81", "u81", "z88", "ctx81", "idx81", "info81",
              "PF89", "PD89", "payload104", "constraints104", "pool_cov104",
              "gate103", "reg90", "registers90"):
    ob131 = globals().get(nm131)
    if hasattr(ob131, "shape"):
        print(nm131, tuple(ob131.shape))
    elif isinstance(ob131, dict):
        print(nm131, "dict", list(ob131)[:8])
    else:
        print(nm131, type(ob131).__name__)

print("\n=== LATENT TENSORS ===")
for nm131 in NAMES:
    if any(s in nm131 for s in ("z_post", "z_prior", "z_in", "z_head")):
        print(nm131, SHAPES[nm131])
```
