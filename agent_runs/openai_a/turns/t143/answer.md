The architecture confirms that dose edits act through the action embedding and recurrent state, while the value head also sees reading history. That makes unwanted feedback in the current graft plausible. The inventory output was truncated, so I’ll retrieve only the causal estimates and transition helpers. This should distinguish an encoding bug from an incorrect memory convention.

```python
import inspect
from pathlib import Path

notes_text = Path("/work/notes.md").read_text()
print("=== INITIAL CAUSAL-EVIDENCE NOTES ===")
print(notes_text[:6500])

print("\n=== RECENT VARIABLE NAMES ===")
print([k for k in list(globals())[-150:] if not k.startswith("_")])

print("\n=== RECONSTRUCTION HELPER DEPENDENCIES ===")
for name in ["rec_memory", "rec_step_with_z", "rec_layer", "rm_set",
             "rm_insulin_atoms", "causal_shift_table"]:
    func = globals().get(name)
    if callable(func) and hasattr(func, "__code__"):
        print(name, "arguments:", func.__code__.co_varnames[:func.__code__.co_argcount],
              "dependencies:", func.__code__.co_names)

print("\n=== TRANSITION HELPER SOURCE, IF RETAINED ===")
func = globals().get("rec_step_with_z")
if callable(func):
    try:
        print(inspect.getsource(func)[:2300])
    except (OSError, TypeError):
        print("Source unavailable; will inspect retained reconstruction configuration.")

print("\n=== RECONSTRUCTION CONFIGURATION ===")
for key, val in list(globals().items()):
    if key.startswith(("rec_", "rm_")) and not callable(val):
        if isinstance(val, (str, int, float, tuple)):
            print(key, repr(val)[:500])
        elif isinstance(val, dict):
            print(key, "keys:", list(val)[:25])
        elif hasattr(val, "shape") and not callable(val.shape):
            print(key, val.shape)
```
