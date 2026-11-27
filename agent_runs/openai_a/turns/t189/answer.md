The last trial produced no accepted candidate: stronger corrections hurt factual fit, while smaller ones did not reliably fix the half-dose response. The previous edit remains in place. I’ll recover the architecture, record-based effect estimates, and prior GPU results before changing anything. A useful next candidate must improve the dose responses without increasing factual NLL by more than 0.005.

```python
from pathlib import Path
import inspect, re, types
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())

restart_notes = Path("/work/notes.md").read_text()
print("=== NOTEBOOK LENGTH / HEADINGS ===", len(restart_notes))
for heading in re.findall(r"^#{1,3} .+$", restart_notes, re.M):
    print(heading)
print("=== EARLY NOTEBOOK: CAUSAL EVIDENCE AND INITIAL VALIDATION ===")
print(restart_notes[:22000])

print("=== KERNEL SCALARS ===")
for key, val in sorted(list(globals().items())):
    if key.startswith("_") or key == "restart_notes":
        continue
    if isinstance(val, (int, float, bool, str, np.number)):
        text = repr(val)
        if len(text) <= 260:
            print(key, "=", text)

print("=== KERNEL FUNCTIONS ===")
for key, val in sorted(list(globals().items())):
    if inspect.isfunction(val) and not key.startswith("_"):
        try:
            print(key, str(inspect.signature(val)))
        except Exception:
            print(key)

print("=== RELEVANT ARRAYS / OBJECTS ===")
pattern = re.compile(r"beta|target|roll|candidate|base|causal|graft|fit|cpu|coef|protect|safe|design|dose|anchor|delta|retained", re.I)
for key, val in sorted(list(globals().items())):
    if not pattern.search(key) or isinstance(val, (types.ModuleType, str)) or inspect.isfunction(val):
        continue
    if hasattr(val, "shape"):
        print(key, type(val).__name__, tuple(val.shape))
    elif isinstance(val, dict):
        print(key, "dict", list(val.keys())[:25])
    elif isinstance(val, (list, tuple)):
        print(key, type(val).__name__, "length", len(val), "preview", repr(val[:3])[:220])

print("=== CURRENT EDITS ===")
print(edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Restart audit; Python 172, forwards 11/50\n"
            "- Run 171 left the beta=-2 rollback unchanged; stronger synthetic contrasts were not accepted.\n"
            "- Recovering architecture, record-based causal evidence, and prior GPU validation before selecting a further edit.\n")
```
