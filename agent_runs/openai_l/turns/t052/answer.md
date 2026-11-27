The architecture shows that doses affect both the latent posterior and the recurrent update. A blanket dose-embedding rescale could therefore damage factual predictions. I’ll recover the validated encoder inputs and the record-based identification results; a stable response to dose innovations would support a targeted, compensated edit.

```python
import inspect
from pathlib import Path

g47 = dict(globals())

print("=== EARLY IDENTIFICATION NOTES ===")
notes47 = Path("/work/notes.md").read_text()
print(notes47[:4400])

print("\n=== VALIDATED FUNCTION DEPENDENCIES ===")
for name in ("contexts37", "start_u43", "affine42", "net42", "ln42"):
    obj = g47.get(name)
    if inspect.isfunction(obj):
        print(name, inspect.signature(obj))
        print("globals:", obj.__code__.co_names)
        print("locals:", obj.__code__.co_varnames)
        print("constants:", repr(obj.__code__.co_consts)[:650])

print("\n=== RECENT RETAINED NAMES ===")
recent47 = [
    name for name in g47
    if not name.startswith("_") and not inspect.ismodule(g47[name])
]
print(", ".join(recent47[-150:]))

print("\n=== RELEVANT ARRAYS AND CONFIGURATIONS ===")
for name in recent47:
    if any(tag in name.lower() for tag in
           ("45", "best44", "kernel", "innovation", "residual", "unitmap", "layout")):
        obj = g47[name]
        if inspect.isfunction(obj):
            continue
        if hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, repr(obj)[:900])
        elif isinstance(obj, (list, tuple, str, int, float)):
            print(name, repr(obj)[:500])

print("\n=== LATENT / PRELUDE TENSORS ===")
for name in NAMES:
    if any(tag in name for tag in ("z_post", "z_prior", "prelude", "z_in", "z_head")):
        print(name, SHAPES[name])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 47\n"
        "Recovered full architecture. Actions affect both latent posterior and "
        "prelude attention; an unconditional embedding rescale is not justified. "
        "Recovering encoder dependencies and early causal identification notes "
        "to construct a residual-dose correction. No edits.\n"
    )
```