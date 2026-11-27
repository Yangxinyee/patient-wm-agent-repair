The last run confirmed the panel layout, but the failed inspection gave no new evidence about drug effects. The weaker insulin edit remains the best validated checkpoint. I’ll recover the architecture and earlier identification work before changing anything; we need to distinguish a poor causal estimate from an edit that fails during rollout.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

rr_notes = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK ===")
print(rr_notes[:16000])

print("\n=== RELEVANT RETAINED OBJECTS ===")
for rr_name, rr_obj in sorted(list(globals().items())):
    if rr_name.startswith("rr_"):
        continue
    if any(s in rr_name.lower() for s in
           ["graft", "robust", "decoder", "gate", "resid", "policy",
            "head", "state", "beta", "rd_", "checkpoint", "register"]):
        if hasattr(rr_obj, "shape"):
            print(rr_name, type(rr_obj).__name__, tuple(rr_obj.shape))
        elif isinstance(rr_obj, dict):
            print(rr_name, "dict keys:", list(rr_obj.keys())[:25])
        elif callable(rr_obj):
            try:
                print(rr_name, inspect.signature(rr_obj))
            except Exception:
                print(rr_name, "callable")
        elif isinstance(rr_obj, (str, int, float, list, tuple)):
            print(rr_name, repr(rr_obj)[:300])

print("\n=== CURRENT EDITS ===")
print(edited())
with Path("/work/notes.md").open("a") as rr_f:
    rr_f.write(
        "\n## Restart recovery; Python 84, forwards 11/50\n"
        "- Retaining the validated beta=-2 insulin graft; no new edits.\n"
        "- Python 83's KeyError was an inspection failure, not treatment evidence.\n"
        "- Recovering exact architecture and causal-identification diagnostics "
        "before another reversible experiment.\n"
    )
```