The recovered estimates agree on the immediate signs: insulin lowers glucose by about 2 units per rate unit, and norepinephrine raises MAP by about 3.17. The longer-horizon response remains uncertain. The output was truncated, so I’ll retrieve only the fixture and lag-analysis objects needed next. Stable lag responses would support a rollout target; inconsistent responses would require revisiting identification.

```python
import numpy as np
import pandas as pd
import inspect
from pathlib import Path

print("=== ACTION FIXTURE ===")
print(action_fixture_repair.columns.tolist())
print(action_fixture_repair.head(2).to_string(index=False))
print("recipes:", atom_recipe_repair)
for r43_drug, r43_atoms in action_atom_vectors_repair.items():
    print("atom bank:", r43_drug, type(r43_atoms).__name__, r43_atoms.shape)
print("numeric specs:")
for r43_key, r43_spec in numeric_specs_repair.items():
    if r43_key[0] in ("insulin", "norepinephrine"):
        print(r43_key, r43_spec)
print("embedding tensors:", [
    (n, SHAPES[n]) for n in NAMES if n.startswith("embed.")
])
print("config:", VOCAB["config"])
print("remaining core tensors:", [
    (n, SHAPES[n]) for n in NAMES
    if n.startswith("core.")
    and not any(s in n for s in ["prelude", "coda", "flow"])
])

print("\n=== CAUSAL VARIABLE NAMES ===")
print([
    n for n in globals()
    if not n.startswith(("r40_", "r41_", "r42_", "r43_"))
    and any(s in n.lower() for s in [
        "dml", "effect", "kernel", "lag", "resid", "innov",
        "moment", "policy", "history", "models", "impulse", "persist"
    ])
])
print("\n=== SMALL DICTIONARY INVENTORY ===")
for r43_name, r43_obj in list(globals().items()):
    if r43_name.startswith(("_", "r40_", "r41_", "r42_", "r43_")):
        continue
    if isinstance(r43_obj, dict) and len(r43_obj) < 15:
        print(r43_name, "keys:", list(r43_obj))

print("\n=== PANEL SAMPLE ===")
print(pan.head(5).to_string())
print("history columns:", HX.columns.tolist())

print("\n=== LONGITUDINAL NOTES ===")
r43_notes = Path("/work/notes.md").read_text()
r43_start = r43_notes.find("## Run 19")
print(r43_notes[r43_start:r43_start + 2100])

with open("/work/notes.md", "a") as r43_file:
    r43_file.write(
        "\n## Run 43\n"
        "Immediate estimates agree approximately with insulin -2 and NE +3.17 per rate. "
        "Recovering exact action fixtures and longitudinal objects with bounded output; "
        "long-horizon causal targets remain provisional. No edits.\n"
    )
```
