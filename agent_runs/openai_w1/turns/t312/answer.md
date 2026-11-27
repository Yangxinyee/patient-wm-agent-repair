The last run found a serious feature mismatch: dose projections were small on the fitting cache but large on reconstructed factual inputs. This could explain the failed edits; it does not establish that the drug-effect estimates were wrong. I’ll recover the architecture and reconstruction code before trying again. Agreement with GPU states and small factual projection errors would support another injection attempt.

```python
from pathlib import Path
import re, inspect, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes273 = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION NOTES ===")
print(notes273[:8500])
print("\n=== DATA DESCRIPTION ===")
print(Path("/data/README.md").read_text())

print("\n=== RECONSTRUCTION / IDENTIFICATION INVENTORY ===")
for name in sorted(list(globals())):
    if re.search(r"(257|258|259|263|266|269|270|safe236|^rr_|^ds_|^rp_)", name):
        obj = globals()[name]
        if callable(obj) and hasattr(obj, "__code__"):
            c = obj.__code__
            print(name, "FUNCTION", c.co_varnames[:c.co_argcount], "GLOBALS", c.co_names)
        elif isinstance(obj, dict):
            print(name, "DICT", list(obj.keys())[:18])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "len", len(obj))
        elif isinstance(obj, (str, int, float)):
            print(name, repr(obj)[:250])

print("\n=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 273\nRestart recovery: safe236 remains installed after both failed recurrent-injection candidates. Run272 exposed a major cached-versus-reconstructed slot-token mismatch. Reading architecture and recovering verified reconstruction and record-identification objects before any new edit. Counts: 273 Python runs, 29 forwards.\n")
```