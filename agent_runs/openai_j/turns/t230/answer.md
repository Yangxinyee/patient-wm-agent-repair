The last result confirms that the local evaluator and dose-contrast arrays are intact. The current edit passes factual validation, but its norepinephrine rollout effects remain below the record-derived estimates. I’ll recover the architecture, causal evidence, and editing helpers before changing weights. Finding weak local dose responses would support a stronger head edit; strong local responses would instead point toward rollout feedback.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes204 = Path("/work/notes.md").read_text()
print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND INITIAL EDITS ===")
print(notes204[:15000])

print("\n=== RELEVANT HELPERS ===")
for name204, obj204 in list(globals().items()):
    if inspect.isfunction(obj204) and any(
        s in name204 for s in ("177", "190", "192", "146", "172", "contrast", "kernel")
    ):
        print(name204, inspect.signature(obj204))
        try:
            print(inspect.getsource(obj204))
        except (OSError, TypeError):
            print("names:", obj204.__code__.co_names)
            print("constants:", repr(obj204.__code__.co_consts)[:2500])

print("\n=== TARGETS ===")
print(Path("/work/kernel_targets.csv").read_text()[:6000])
print("\n=== RELEVANT PERSISTENT VARIABLES ===")
for name204, obj204 in list(globals().items()):
    if any(s in name204 for s in ("head177", "map192", "cf172", "map190")):
        if inspect.isfunction(obj204):
            continue
        if isinstance(obj204, dict):
            print(name204, "dict keys:", list(obj204)[:30])
        else:
            print(name204, type(obj204).__name__, getattr(obj204, "shape", ""),
                  repr(obj204)[:450] if not hasattr(obj204, "shape") else "")
print("\nActive edits:", edited())
```
