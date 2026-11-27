The architecture confirms that dose changes enter the posterior and recurrent state, while generated readings feed back through both the state and numeric registers. The inspection output was truncated before the retained helpers appeared. I’ll retrieve a compact inventory so I can validate the full transition rather than guess its implementation.

```python
import inspect
import re
from pathlib import Path

r98_items = list(globals().items())
print("=== RETAINED USER FUNCTIONS ===")
for r98_name, r98_obj in sorted(r98_items):
    if inspect.isfunction(r98_obj) and getattr(r98_obj, "__module__", "") == "__main__":
        try:
            print(r98_name, inspect.signature(r98_obj))
        except Exception:
            print(r98_name)

print("\n=== ARRAY / TABLE / DICTIONARY NAMES ===")
r98_inventory = []
for r98_name, r98_obj in sorted(r98_items):
    if r98_name.startswith("_") or r98_name.startswith(("r97_", "r98_")):
        continue
    if hasattr(r98_obj, "shape"):
        r98_inventory.append(f"{r98_name}:{tuple(r98_obj.shape)}")
    elif isinstance(r98_obj, dict):
        r98_inventory.append(f"{r98_name}:dict({len(r98_obj)})")
    elif isinstance(r98_obj, (list, tuple)) and len(r98_obj) > 0:
        r98_inventory.append(f"{r98_name}:{type(r98_obj).__name__}({len(r98_obj)})")
print(" | ".join(r98_inventory))

print("\n=== NOTEBOOK SECTION HEADINGS ===")
r98_text = Path("/work/notes.md").read_text()
print("\n".join(re.findall(r"^## .*$", r98_text, flags=re.M)))

with Path("/work/notes.md").open("a") as r98_f:
    r98_f.write(
        "\n## Compact recovery; Python 98, forward 11/50\n"
        "- Architecture recovered; previous inspection output was truncated.\n"
        "- Retrieving retained helper names before validating recurrent transitions.\n"
        "- No weight changes.\n"
    )
```
