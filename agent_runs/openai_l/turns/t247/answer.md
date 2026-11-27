The nonlinear candidate failed the factual limit and has been rolled back. I suspect its dose features did not generalize beyond the cached stays. I’ll recover the architecture and fitting code before testing that hypothesis; independent factual coverage must confirm any improvement.

```python
from pathlib import Path
import re, inspect, dis, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes222 = Path("/work/notes.md").read_text()
sections222 = re.split(r"(?=^## )", notes222, flags=re.M)
print("\n=== EARLY CAUSAL-EVIDENCE NOTES ===")
for section in sections222:
    if re.search(r"^## .*?(?:Run[s]? (?:[1-9]\b|1[0-9]\b|2[0-1]\b)|causal|kernel|identif)", section, re.I):
        print(section[:3500])
print("\n=== RECENT LIVE OBJECTS ===")
for name in sorted(list(globals())):
    match = re.search(r"(\d+)$", name)
    if match and 200 <= int(match.group(1)) <= 219:
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape, obj.dtype)
        elif callable(obj):
            try:
                print(name, "function", inspect.signature(obj),
                      "locals", obj.__code__.co_varnames,
                      "references", obj.__code__.co_names)
            except Exception:
                print(name, type(obj).__name__)
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj.keys())[:25])
        else:
            print(name, type(obj).__name__, str(obj)[:180])
print("\n=== CURRENT EDITS ===", edited())
print("\n=== STATE SOURCE IDENTIFIERS ===")
for path in sorted(Path("/work").glob("fwd_*/states_*.npz")):
    with np.load(path) as data:
        ids = np.unique(data["stay_id"])
        print(str(path), len(ids), ids[:12].tolist())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 222 / independent-coverage diagnosis\n"
            "Recovered architecture and fitting interfaces after rollback. "
            "Current five-tensor conservative edit remains staged; no new weights saved. "
            "Investigating whether nonlinear dose-feature generalization caused the "
            "offline/GPU factual discrepancy before attempting another candidate.\n")
```