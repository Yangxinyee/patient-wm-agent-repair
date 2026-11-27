The earlier tests show that erasing dose-bin distinctions barely changed the rollouts. I’ll finish recovering the selective-edit results and inspect where an edit could alter recurrent state, not just the reading head. A useful state correction should strengthen later drug responses without shifting factual readings substantially.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== NOTEBOOK: SELECTIVE EDIT RESULTS ===")
print(Path("/work/notes.md").read_text()[12800:18200])

print("\n=== TRANSFORM HELPER SIGNATURES ===")
for _name in (
    "replay_ops", "apply_lowrank_left_transform",
    "apply_lowrank_right_transform", "effective_transform",
    "freeze_ops", "state_dose_coefficients"
):
    _f = globals().get(_name)
    if callable(_f):
        print(_name, inspect.signature(_f))

print("\n=== CORE OUTPUT / CODA TENSORS ===")
for _n in NAMES:
    if any(_term in _n for _term in ("core.c_out", "core.coda", "core.b_in")):
        print(_n, SHAPES[_n])

print("\n=== DESIGN DICTIONARIES (KEYS ONLY) ===")
for _name, _value in list(globals().items()):
    if (not _name.startswith("_") and isinstance(_value, dict)
            and _name not in ("SHAPES", "VOCAB")
            and any(_term in _name.lower() for _term in (
                "head", "design", "candidate", "fit", "sensor", "latent", "controlled"
            ))):
        print(_name, list(_value.keys())[:16])

print("\n=== DOSE DIRECTION STRUCTURE ===")
for _name, _value in dose_directions.items():
    if isinstance(_value, dict):
        print(_name, {k: (np.shape(v) if isinstance(v, np.ndarray) else type(v).__name__)
                      for k, v in _value.items()})
    else:
        print(_name, type(_value).__name__, np.shape(_value))

with open("/work/notes.md", "a") as _notes:
    _notes.write(
        "\n## Python run 65\n"
        "- Reviewing prior selective-head tests and possible recurrent-state correction locations.\n"
        "- No edits changed; numeric gain 4 remains installed; forward passes used: 25.\n"
    )
```