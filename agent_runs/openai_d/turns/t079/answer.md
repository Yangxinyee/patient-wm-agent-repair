The reported error is in the inventory display, not the causal analysis, and no weights changed. I’ll avoid inspecting shapes and retrieve only the response kernels and variable names. This should recover the information needed to choose a dose-sensitive pathway without repeating the failed display.

```python
import json
from pathlib import Path

r71_text = Path("/work/notes.md").read_text()
r71_start = r71_text.find("## Run 5\n")
r71_start = r71_text.find("{", r71_start)
r71_lp, _ = json.JSONDecoder().raw_decode(r71_text[r71_start:])

print("=== RESPONSE KERNELS ===")
for r71_drug in ("insulin", "norepinephrine"):
    r71_info = r71_lp.get(r71_drug, {})
    print(r71_drug, "kernel:", r71_info.get("kernel"))
    print(r71_drug, "geometric:", r71_info.get("geometric"))

print("\n=== NEXT NOTES ===")
r71_start = r71_text.find("## Run 6")
if r71_start >= 0:
    print(r71_text[r71_start:r71_start + 2600])

print("\n=== VARIABLE NAMES AND TYPES, NEWEST FIRST ===")
r71_inventory = [
    n + ":" + type(v).__name__
    for n, v in reversed(list(globals().items()))
    if not n.startswith(("_", "r71", "rec"))
]
print("; ".join(r71_inventory)[:4200])

print("\n=== FUNCTION NAMES ===")
print(", ".join(
    n for n, v in list(globals().items())
    if hasattr(v, "__code__") and not n.startswith("_")
)[:1200])
print("\nCurrent edits:", edited())
```