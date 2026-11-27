The value register matches exactly, but the dose-encoder mismatch is too large to trust an edit. I’ll recover the validated encoder and the record-based effect estimates before proceeding. Agreement with the exported states would confirm the reconstruction; persistent disagreement would refute it.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK: IDENTIFICATION EVIDENCE ===")
notes68 = Path("/work/notes.md").read_text()
print(notes68[:11500])

print("\n=== RETAINED FUNCTIONS ===")
for name68, obj68 in list(globals().items()):
    if inspect.isfunction(obj68) and getattr(obj68, "__module__", "") == "__main__":
        try:
            sig68 = str(inspect.signature(obj68))
        except Exception:
            sig68 = "?"
        print(name68, sig68)

print("\n=== RECENT VARIABLES AND RELEVANT EARLIER OBJECTS ===")
for name68, obj68 in list(globals().items()):
    relevant68 = (
        name68.endswith(("60", "61", "62", "63", "64", "65", "66", "67"))
        or any(s68 in name68.lower() for s68 in
               ("kernel", "causal", "policy", "r32_best", "train_ids", "action_candidates"))
    )
    if relevant68 and not name68.startswith("_") and not inspect.isfunction(obj68):
        if hasattr(obj68, "shape"):
            print(name68, type(obj68).__name__, tuple(obj68.shape))
            if hasattr(obj68, "columns"):
                print("  columns:", list(obj68.columns))
        elif isinstance(obj68, dict):
            print(name68, "dict keys:", list(obj68)[:35])
        elif isinstance(obj68, (tuple, list)):
            print(name68, type(obj68).__name__, "length", len(obj68))
        elif isinstance(obj68, (str, int, float, bool, type(None))):
            print(name68, repr(obj68)[:180])

print("\nCurrent edits:", edited())
with Path("/work/notes.md").open("a") as f68:
    f68.write(
        "\n## Run 68 / recovery\n"
        "Run 67's generic register is exact, but its explicit action encoder check "
        "has RMS 1.0097 and cannot support an edit. Recovering the previously "
        "validated encoder, recurrence, decoder, and record-based causal estimates. "
        "No edits; forward passes 4/50.\n"
    )
```