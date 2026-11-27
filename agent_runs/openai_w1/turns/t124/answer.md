The last test rules out rate-unit IDs as the reading-encoder problem; the drug-only path is already accurate. I’ll recover the architecture, earlier effect estimates, and validated helpers before editing anything. The reconstruction must match exported states before I trust an edit derived from it.

```python
from pathlib import Path
import inspect, re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RELEVANT HELPERS ===")
for _name, _obj in sorted(list(globals().items())):
    if callable(_obj) and not isinstance(_obj, type) and (
        _name.startswith(("rr_", "r8", "r9", "r10", "r11")) or
        any(s in _name.lower() for s in ("atom", "posterior", "kernel", "innovation"))
    ):
        try:
            _sig = str(inspect.signature(_obj))
        except Exception:
            _sig = "?"
        print(_name, _sig)

print("\n=== FRESH CONSTANTS ===")
for _name in ("rr_units", "rr_drugs", "rr_sensors"):
    if _name in globals():
        print(_name, globals()[_name])

print("\n=== EARLY NOTEBOOK EVIDENCE ===")
_text = Path("/work/notes.md").read_text()
_sections = re.split(r"(?m)(?=^## )", _text)
_keywords = ("deconvol", "causal", "innovation", "dose_edit", "rollout", "effect estimate", "kernel")
_selected = [s for s in _sections if any(k in s.lower() for k in _keywords)]
print("\n".join(s[:1600] for s in _selected[:14]))

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 113\nRecovered architecture, identification notes, and helper inventory after restart. "
             "Run 112 ruled out rate-unit alternatives; reading reconstruction remains unvalidated. "
             "No weights edited. Counts: 113 Python runs, 6 forward passes.\n")
```