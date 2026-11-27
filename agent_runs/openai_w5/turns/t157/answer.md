The latest pass saved paired states, but did not establish factual fit or causal accuracy. I’ll reread the architecture and earlier evidence before auditing dose transport. Correct timing and limited disturbance to ordinary states would support this approach; either failure would argue against it.

```python
from pathlib import Path as _R136Path

print("=== ARCHITECTURE ===")
print(_R136Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(_R136Path("/data/README.md").read_text())

_r136_notes = _R136Path("/work/notes.md").read_text()
print("=== EARLIER NOTEBOOK EVIDENCE ===")
print(_r136_notes[:18000])
print("=== PERSISTENT VARIABLE NAMES ===")
print(", ".join(sorted(
    k for k in globals()
    if not k.startswith("_") and k not in {"In", "Out", "get_ipython", "exit", "quit"}
)))
print("=== CURRENT EDITS ===")
print(edited())
```
