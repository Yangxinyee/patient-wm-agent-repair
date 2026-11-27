```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

rc_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK / CAUSAL IDENTIFICATION ===")
print(rc_notes[:17000])
print("\n=== TARGETS AND FORWARD CHECKPOINTS ===")
rc_lines = rc_notes.splitlines()
rc_matches = [
    i for i, s in enumerate(rc_lines)
    if any(k in s.lower() for k in [
        "true effect", "target effect", "causal target", "target:", "targets:",
        "fwd_", "forward pass", "rollout shift", "dose-edit", "dose_edit",
        "instrument", "randomiz"
    ])
]
for i in rc_matches[-65:]:
    print("\n".join(rc_lines[max(0, i-1):min(len(rc_lines), i+4)]))

print("\n=== AVAILABLE FUNCTIONS ===")
for rc_name, rc_obj in list(globals().items()):
    if inspect.isfunction(rc_obj) and not rc_name.startswith("rc_"):
        try:
            print(rc_name, inspect.signature(rc_obj))
        except Exception:
            print(rc_name)

print("\n=== RECENT CANDIDATE OBJECTS ===")
for rc_name, rc_obj in list(globals().items()):
    if rc_name.startswith(("r230", "r231", "r232")):
        if hasattr(rc_obj, "shape"):
            print(rc_name, type(rc_obj).__name__, rc_obj.shape)
        elif isinstance(rc_obj, dict):
            print(rc_name, "dict keys:", list(rc_obj.keys())[:35])
        elif isinstance(rc_obj, (int, float, str, list, tuple)):
            print(rc_name, repr(rc_obj)[:600])
print("\nCURRENT EDITS:", edited())

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 233: restart recovery\n"
            "Recovered architecture, causal-target history, forward checkpoints, "
            "and candidate functions. Current .94 insulin / .40 NE trial retained; "
            "no weights changed. Its cached tune100 NLL cost is 0.004616, so both "
            "actual factual fit and rollout effects still need validation.\n")
```