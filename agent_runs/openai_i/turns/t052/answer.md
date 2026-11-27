The reconstructed start-of-infusion inputs are close, but the END-event encoding remains unresolved. No repair has been tested yet. I’ll recover the architecture and earlier causal estimates, then decide whether we can build a correction without solving that encoding. A held-out, dose-sensitive correction with little factual loss would support that approach.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("\n=== EARLIER NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:11000])

print("\n=== RETAINED VARIABLES ===")
_inventory = []
for _name, _obj in list(globals().items()):
    if _name.startswith("_"):
        continue
    if hasattr(_obj, "shape"):
        _inventory.append(f"{_name}: {type(_obj).__name__} {tuple(_obj.shape)}")
    elif isinstance(_obj, dict):
        _inventory.append(f"{_name}: dict({len(_obj)}) keys={list(_obj)[:7]}")
    elif isinstance(_obj, (list, tuple)):
        _inventory.append(f"{_name}: {type(_obj).__name__}({len(_obj)})")
    elif callable(_obj) and getattr(_obj, "__module__", "") == "__main__":
        try:
            _inventory.append(f"{_name}{inspect.signature(_obj)}")
        except Exception:
            _inventory.append(f"{_name}: callable")
print("\n".join(_inventory))

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 45; forward 6 — recovery\n"
             "- No edits. Recovering architecture, record-based effect estimates, and retained helpers.\n"
             "- Start encoding is close; END encoding is unresolved. Avoid further categorical searches unless needed for a validated correction.\n")
```
