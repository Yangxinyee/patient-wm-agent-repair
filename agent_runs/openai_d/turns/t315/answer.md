The reduced edit increased factual NLL by about **0.00457 nats**, just inside the 0.005 limit on this cohort. That is encouraging, but not enough to establish safety across stays. I’ll recover the architecture, record-based effect estimates, and saved validation setup before testing further. Broader factual validation below the limit—and preserved dose responses—would support keeping this candidate.

```python
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: CAUSAL FINDINGS AND EDIT DESIGN ===")
print(_notes[:18000])
print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in _notes.splitlines() if line.startswith("#")))

print("\n=== AVAILABLE KERNEL VARIABLES ===")
print(", ".join(sorted(
    k for k in list(globals())
    if not k.startswith("_") and k not in {"In", "Out", "get_ipython", "exit", "quit"}
)))
print("\n=== CURRENT EDITS ===")
print(edited())
```