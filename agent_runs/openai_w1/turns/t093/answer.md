The inspection stopped because the numeric vocabulary is a list, not a dictionary. That error tells us nothing about the model, and no weights changed. I’ll correct the inspection and recover the aligned encoder inputs. Those inputs will let us test the unit-ID hypothesis rather than guess.

```python
import inspect
import numpy as np
import torch

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 83–84\n"
        "Run 83 stopped while treating the list-valued numeric vocabulary as a "
        "dictionary. No experiment or weight edit occurred. Run 84 corrects "
        "inspection and recovers encoder inputs. "
        "Counts: 84 Python runs, 6 forward passes.\n"
    )

print("=== NUMERIC VOCAB STRUCTURE ===")
print(type(r82_vocab["numeric"]).__name__, "length", len(r82_vocab["numeric"]))
print(json.dumps(r82_vocab["numeric"], indent=2)[:4200])

def r84_describe(x):
    if callable(x):
        try:
            return "callable " + str(inspect.signature(x))
        except Exception:
            return "callable"
    if hasattr(x, "shape"):
        return f"{type(x).__name__} shape={tuple(x.shape)}"
    if isinstance(x, dict):
        return f"dict({len(x)}) keys={list(x)[:10]}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}({len(x)})"
    return type(x).__name__

print("\n=== CORE RECONSTRUCTION OBJECTS ===")
for r84_name in sorted(list(globals())):
    if r84_name.startswith("r33_"):
        print(r84_name, r84_describe(globals()[r84_name]))

print("\n=== RECENT ALIGNMENT AND ENCODER OBJECTS ===")
r84_recent = []
for r84_name in sorted(list(globals())):
    if r84_name.startswith(("r74_", "r78_", "r79_", "r80_", "r81_")):
        r84_obj = globals()[r84_name]
        if callable(r84_obj) or isinstance(r84_obj, dict) or (
            hasattr(r84_obj, "shape") and len(r84_obj.shape) >= 2
        ):
            r84_recent.append((r84_name, r84_describe(r84_obj)))
for r84_item in r84_recent[:55]:
    print(*r84_item)
print("recent objects shown:", min(55, len(r84_recent)), "of", len(r84_recent))

print("\n=== EARLY EXPERIMENT SUMMARY ===")
r84_notes = Path("/work/notes.md").read_text()
print(r84_notes[:5200])
```