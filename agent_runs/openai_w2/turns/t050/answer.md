The earlier statistical fits and model probes are still in memory. That lets me reuse the evidence rather than repeat expensive work. I’ll recover the prior edit diagnostics and numeric normalisation next. If the earlier probe found a usable insulin action direction, we can target it while avoiding the head-only edits already rejected for factual cost.

```python
import inspect
from pathlib import Path

print("=== AVAILABLE FUNCTIONS ===")
for key, val in list(globals().items()):
    if inspect.isfunction(val) and not key.startswith("_"):
        try:
            print(key, inspect.signature(val))
        except Exception:
            print(key)

print("=== REMAINING VARIABLES ===")
items = list(globals().items())
start = next((i for i, (key, val) in enumerate(items) if key == "head_all_check"), len(items))
for key, val in items[start:]:
    if key.startswith("_") or key in {"val", "items", "text", "inventory", "notes_text"}:
        continue
    shape = getattr(val, "shape", None)
    if isinstance(shape, (tuple, list)):
        info = str(tuple(shape))
    elif isinstance(val, dict):
        info = "keys=" + ",".join(map(str, list(val)[:8]))
    elif isinstance(val, (list, tuple)):
        info = "len=" + str(len(val))
    elif isinstance(val, (str, int, float)):
        info = repr(val)[:100]
    else:
        info = ""
    print(key, type(val).__name__, info)

print("=== SELECTED NOTEBOOK SECTIONS ===")
text = Path("/work/notes.md").read_text()
for heading in ["## Run 11:", "## Run 12:", "## Runs 25", "## Run 27:", "## Run 30:", "## Run 32:", "## Run 33:"]:
    pos = text.find(heading)
    if pos >= 0:
        end = text.find("\n## ", pos + 4)
        section = text[pos:end if end >= 0 else len(text)]
        print(section[:1100])

print("=== NUMERIC NORMALISATION ===")
for key, spec in numeric_specs.items():
    print(key, {k: v for k, v in spec.items() if k != "fine_bin_edges"})
print("=== FORWARD ARTIFACTS ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])
```
