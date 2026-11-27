The validated checkpoint was restored exactly. The recovery output was truncated, so it has not yet resolved the causal targets or how the gate is constructed. I’ll locate those specific notes and helpers before changing anything; they must support a reproducible edit rather than another proxy-only trial.

```python
import inspect
import re
from pathlib import Path

recovery_globals = dict(globals())
notes_text = Path("/work/notes.md").read_text()
note_sections = re.split(r"(?m)(?=^## )", notes_text)

print("=== RELEVANT NOTEBOOK HEADINGS ===")
heading_terms = (
    "identif", "causal", "fuzzy", "discontinu", "instrument", "persistence",
    "norepinephrine", "negative", "null", "target", "graft", "activity", "gate"
)
headings = [
    s.splitlines()[0] for s in note_sections if s.strip()
    and any(t in s.splitlines()[0].lower() for t in heading_terms)
]
print("\n".join(headings)[:3500])

print("\n=== PYTHON HELPERS ===")
helpers = []
for name, obj in recovery_globals.items():
    if name.startswith("_") or not inspect.isfunction(obj):
        continue
    try:
        sig = str(inspect.signature(obj))
    except Exception:
        sig = "(signature unavailable)"
    helpers.append(f"{name}{sig}")
print("\n".join(sorted(helpers))[:2800])

print("\n=== RELEVANT VARIABLES ===")
variable_terms = (
    "activity", "gate", "r3", "beta", "persist", "rd_", "target",
    "tune_ids", "tuning_ids", "cache", "checkpoint"
)
lines = []
for name in sorted(recovery_globals):
    if name.startswith("_") or not any(t in name.lower() for t in variable_terms):
        continue
    obj = recovery_globals[name]
    if callable(obj):
        continue
    if hasattr(obj, "shape"):
        desc = f"{type(obj).__name__} {obj.shape}"
    elif isinstance(obj, dict):
        desc = f"dict {list(obj)[:7]}"
    elif isinstance(obj, (list, tuple)):
        desc = f"{type(obj).__name__} len={len(obj)} first={repr(obj[:3])[:90]}"
    else:
        desc = repr(obj)[:130]
    lines.append(f"{name}: {desc}")
print("\n".join(lines)[:2500])
```