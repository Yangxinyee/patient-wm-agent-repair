The records support substantial effects in opposite directions: insulin lowers later glucose, while norepinephrine raises later MAP. The original rollouts largely miss both. I’ll recover which action-path edits were already tested so I can target an untested pathway rather than repeat failed broad amplifications.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== NOTEBOOK: EARLIER EDIT TESTS ===")
print(Path("/work/notes.md").read_text()[6500:12800])

print("\n=== SAVED CANDIDATES ===")
print(list(saved_candidates.keys()))
for _name in ("A_bin_erasure", "B_posterior_action_erasure", "C_numeric_x4"):
    if _name in saved_candidates:
        print(_name, saved_candidates[_name])

print("\n=== TRANSFORM HELPERS ===")
for _name in (
    "replay_ops", "apply_lowrank_left_transform",
    "apply_lowrank_right_transform", "effective_transform",
    "freeze_ops", "state_dose_coefficients"
):
    _f = globals().get(_name)
    if callable(_f):
        print(_name, inspect.signature(_f), inspect.getdoc(_f))

print("\n=== REMAINING SMALL DICTIONARY INVENTORY ===")
for _name, _value in list(globals().items()):
    if (not _name.startswith("_") and isinstance(_value, dict)
            and _name not in ("SHAPES", "VOCAB")
            and any(_term in _name.lower() for _term in (
                "head", "design", "candidate", "fit", "sensor", "latent", "controlled"
            ))):
        print(_name, list(_value.keys())[:20])

with open("/work/notes.md", "a") as _notes:
    _notes.write(
        "\n## Python run 64\n"
        "- Recovered training-only causal identification: insulin about -2 and NE about +3 per rate unit immediately, with persistence.\n"
        "- Reviewing earlier action-path tests and existing transform helpers before another selective experiment.\n"
        "- Numeric gain 4 retained; forward passes used: 25.\n"
    )
```