The second inspection overflowed because a helper’s default argument contains the weight dictionary. It produced no new evidence about the model. I’ll print only parameter names and bounded summaries, so we can locate the reconstruction code and test whether it also reproduces the next hidden state.

```python
import inspect
import re
from pathlib import Path

r99_items = list(globals().items())

r99_functions = []
for r99_name, r99_obj in sorted(r99_items):
    if inspect.isfunction(r99_obj) and getattr(r99_obj, "__module__", "") == "__main__":
        try:
            r99_params = ",".join(inspect.signature(r99_obj).parameters)
        except Exception:
            r99_params = "?"
        r99_functions.append(f"{r99_name}({r99_params})")
print("=== FUNCTIONS ===")
print("\n".join(r99_functions)[:2600])

r99_objects = []
for r99_name, r99_obj in sorted(r99_items):
    if r99_name.startswith("__") or r99_name.startswith(("r97_", "r98_", "r99_")):
        continue
    if hasattr(r99_obj, "shape"):
        r99_objects.append(f"{r99_name}:{tuple(r99_obj.shape)}")
    elif isinstance(r99_obj, dict):
        r99_objects.append(f"{r99_name}:dict[{len(r99_obj)}]")
    elif isinstance(r99_obj, (list, tuple)):
        r99_objects.append(f"{r99_name}:{type(r99_obj).__name__}[{len(r99_obj)}]")
r99_object_text = " | ".join(r99_objects)
print("\n=== OBJECT INVENTORY (FIRST PART) ===")
print(r99_object_text[:3800])
print("Inventory total characters:", len(r99_object_text))

r99_sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
r99_causal_sections = [
    s for s in r99_sections
    if re.search(r"(?i)\bRD\b|regression discontinu|threshold|persistence", s)
    and not re.search(r"(?i)^##.*recovery", s)
]
print("\n=== CAUSAL EVIDENCE EXCERPTS ===")
print("\n".join(r99_causal_sections)[:2300])

with Path("/work/notes.md").open("a") as r99_f:
    r99_f.write(
        "\n## Bounded recovery; Python 99, forward 11/50\n"
        "- Avoiding large tensor-valued function defaults in inspection output.\n"
        "- No new scientific conclusions or weight changes.\n"
    )
```
