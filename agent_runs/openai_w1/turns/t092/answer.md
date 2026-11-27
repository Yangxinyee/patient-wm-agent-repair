The documentation confirms that the two glucose sensors share a unit, as do the two MAP sensors. However, the vocabulary does not specify the categorical unit IDs, so the helper’s assumed IDs still need checking. I’ll recover the encoder helpers and aligned arrays for a controlled unit-ID test. A corrected ID should remove the reading mismatch without disturbing the exact END reconstruction.

```python
import inspect
import numpy as np
import torch

print("=== CONFIG AND NUMERIC VOCAB ===")
print(json.dumps(r82_vocab["config"], indent=2))
print("codes:", r82_vocab["codes"])
print("numeric_note:", r82_vocab.get("numeric_note"))
for r83_code in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
                 "insulin", "norepinephrine"]:
    for r83_attr, r83_table in r82_vocab["numeric"].items():
        if isinstance(r83_table, dict) and r83_code in r83_table:
            print(r83_attr, r83_code, r83_table[r83_code])
    if r83_code in r82_vocab["numeric"]:
        print(r83_code, r82_vocab["numeric"][r83_code])

def r83_describe(x):
    if callable(x):
        try:
            return "callable " + str(inspect.signature(x))
        except Exception:
            return "callable"
    if hasattr(x, "shape"):
        return f"{type(x).__name__} shape={tuple(x.shape)}"
    if isinstance(x, dict):
        return f"dict({len(x)}) keys={list(x)[:12]}"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}({len(x)})"
    return type(x).__name__

print("\n=== CORE RECONSTRUCTION OBJECTS ===")
for r83_name in sorted(list(globals())):
    if r83_name.startswith("r33_"):
        print(r83_name, r83_describe(globals()[r83_name]))

print("\n=== RECENT ALIGNMENT AND ENCODER OBJECTS ===")
for r83_name in sorted(list(globals())):
    if r83_name.startswith(("r74_", "r78_", "r79_", "r80_", "r81_")):
        r83_obj = globals()[r83_name]
        if callable(r83_obj) or isinstance(r83_obj, dict) or (
            hasattr(r83_obj, "shape") and len(r83_obj.shape) >= 2
        ):
            print(r83_name, r83_describe(r83_obj))

print("\n=== EARLY EXPERIMENT SUMMARY ===")
r83_notes = Path("/work/notes.md").read_text()
print(r83_notes[:4600])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 83\n"
        "Architecture confirms measurement units but vocabulary lacks categorical "
        "ID labels. Recovering exact helpers and aligned arrays for an isolated "
        "unit-ID test; no posterior calibration or edit is accepted yet. "
        "Counts: 83 Python runs, 6 forward passes.\n"
    )
```