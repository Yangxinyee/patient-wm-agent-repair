The notebook reports similar history-adjusted insulin effects across both glucose sensors and positive norepinephrine effects on both MAP sensors. The original rollouts respond much less, but that alone does not establish a safe weight edit. I’ll identify the retained reconstruction and causal-analysis helpers so I can validate them without repeating earlier work.

```python
import inspect
from pathlib import Path

r21_items = list(globals().items())
print("=== NOTEBOOK FUNCTIONS ===")
for r21_name, r21_obj in r21_items:
    if inspect.isfunction(r21_obj) and getattr(r21_obj, "__module__", "") == "__main__":
        if not r21_name.startswith("r20_"):
            print(r21_name, str(inspect.signature(r21_obj)),
                  "calls/fields:", ",".join(r21_obj.__code__.co_names))

print("\n=== ARRAYS AND TABLES ===")
for r21_name, r21_obj in r21_items:
    if r21_name.startswith(("r20_", "r21_", "_")):
        continue
    if hasattr(r21_obj, "shape") and not inspect.isclass(r21_obj):
        print(r21_name, type(r21_obj).__name__, tuple(r21_obj.shape))

print("\n=== OTHER RETAINED OBJECTS ===")
for r21_name, r21_obj in r21_items:
    if r21_name.startswith(("r20_", "r21_", "_")):
        continue
    if isinstance(r21_obj, dict):
        print(r21_name, "dict", len(r21_obj), "keys:", list(r21_obj)[:4])
    elif isinstance(r21_obj, (list, tuple)):
        print(r21_name, type(r21_obj).__name__, len(r21_obj))
    elif isinstance(r21_obj, (int, float, bool)):
        print(r21_name, repr(r21_obj))

print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

with Path("/work/notes.md").open("a") as r21_file:
    r21_file.write(
        "\n## Run 21: retained-helper inventory\n"
        "- Earlier cross-fitted immediate effects agree across glucose sensors "
        "and indicate positive MAP effects, but identification and dynamic "
        "extrapolation still require checks.\n"
        "- Recovering exact helper names and data structures; run 20 output "
        "was truncated before this inventory.\n"
        "- Python runs: 21; forward passes: 4; no weights edited.\n"
    )
```