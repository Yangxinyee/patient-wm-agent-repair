The kernel still has the training panel and prior analysis objects. The inspection stopped on a closed NPZ handle, not a model failure. I’ll inspect the posterior helpers and recent arrays with guarded reads, then test zero-action inputs. Numerical agreement would confirm that correction; residual errors would identify another missing convention.

```python
import dis
import inspect
import re
import torch
import numpy as np
from pathlib import Path

print("=== FORWARD FILES ===")
for p in sorted(Path("/work").glob("fwd_*/*")):
    print(str(p))

print("\n=== RECENT / LATENT ARRAYS ===")
for name, obj in sorted(list(globals().items())):
    if not (name.startswith("latent") or re.search(r"(6[5-9]|7[0-3])$", name)):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape), str(obj.dtype))
    elif isinstance(obj, dict):
        print(name, "keys", list(obj)[:12])
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, "length", len(obj))
    elif isinstance(obj, str) and len(obj) > 150:
        print(name, "source-string length", len(obj))

print("\n=== POSTERIOR HELPER IMPLEMENTATIONS ===")
for name in ("latent_context", "latent_post_mean", "latent_hidden"):
    obj = globals()[name]
    print("\nFUNCTION", name, "REFERENCES", obj.__code__.co_names)
    try:
        print(inspect.getsource(obj))
    except (OSError, TypeError):
        for ins in dis.get_instructions(obj):
            print(ins.offset, ins.opname, ins.argrepr[:120])

print("\n=== OTHER HELPER REFERENCES ===")
for name in ("encode_grouped65", "post_mean59", "mlp65",
             "complete_atom57", "head_exact"):
    obj = globals()[name]
    print(name, obj.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 76; forward 6 — guarded helper recovery\n"
        "- Prior inspection encountered a closed NPZ handle; no model operation failed.\n"
        "- Training panels and statistical helpers remain in memory. Inspecting "
        "posterior arrays and exact helper implementations to test absent-action u=0.\n"
        "- No weights changed.\n"
    )
```