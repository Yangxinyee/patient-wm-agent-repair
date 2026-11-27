The stronger balanced edit failed the factual-fit limit, so candidate58 remains restored. The norepinephrine table shows how dosing depends on MAP, not what the drug causes. I’ll recover the architecture and earlier identification evidence before choosing another edit. We need treatment effects supported by dose variation among comparable histories, and a repair that leaves factual predictions intact.

```python
from pathlib import Path
import inspect
import numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes65 = Path("/work/notes.md").read_text()
print("\nEARLY NOTEBOOK")
print(notes65[:14500])

print("\nRETAINED OBJECT INVENTORY")
inventory65 = []
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if inspect.isfunction(obj):
        try:
            desc = str(inspect.signature(obj))
        except Exception:
            desc = "(signature unavailable)"
        inventory65.append((name, "function", desc))
    elif isinstance(obj, (np.ndarray, dict, list, tuple)) or type(obj).__name__ in ("DataFrame", "Series"):
        if isinstance(obj, dict):
            desc = f"len={len(obj)}, keys={list(obj)[:8]}"
        elif hasattr(obj, "shape"):
            desc = f"shape={obj.shape}"
        else:
            desc = f"len={len(obj)}"
        inventory65.append((name, type(obj).__name__, desc))
for row in inventory65:
    print(*row, sep=" | ")

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 65: second restart recovery\n"
            "Candidate63 was rejected for factual degradation; candidate58 remains restored. "
            "The NE dose/MAP table is observational policy evidence, not a causal estimate. "
            "Reading architecture, early identification notes, and retained objects before further edits. "
            "No weights changed. Python runs: 65; forwards: 8.\n")
```