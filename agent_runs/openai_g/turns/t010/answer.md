The richer history controls left norepinephrine’s one-hour association near zero. Its apparent delayed association largely disappeared after the first four hours, suggesting early-stay confounding rather than a stable effect.

I’ll inspect the architecture and surviving kernel objects before choosing an edit. Near-zero effects remain the working hypothesis; consistent later-hour effects would challenge it. Any edit must also preserve factual likelihood.

```python
from pathlib import Path
import numpy as np
import json
import types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

print("\n=== EARLIER NOTEBOOK CONTEXT ===")
print(Path("/work/notes.md").read_text()[:6500])

print("\n=== TENSOR NAMES AND SHAPES ===")
for name in NAMES:
    print(name, SHAPES[name])

print("\n=== VOCAB STRUCTURE ===")
for key, value in VOCAB.items():
    if isinstance(value, dict):
        print(key, "keys:", list(value.keys()))
        print(json.dumps(value, default=str)[:1000])
    else:
        print(key, str(value)[:700])

print("\n=== SURVIVING ARRAYS / TABLES ===")
for name, obj in list(globals().items()):
    if not name.startswith("_") and hasattr(obj, "shape"):
        print(name, type(obj).__name__, obj.shape, getattr(obj, "dtype", ""))

print("\n=== SURVIVING USER FUNCTIONS ===")
for name, obj in list(globals().items()):
    if isinstance(obj, types.FunctionType) and obj.__module__ == "__main__":
        print(name)

print("\nCurrent edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Restart inspection\n"
        "Run 8's richer-history NE estimates are near zero at one hour. "
        "The twelve-hour association is concentrated before hour 4 and is much "
        "smaller thereafter; this suggests early-stay residual confounding. "
        "Working hypothesis: both drugs have little or no causal effect, "
        "subject to further later-hour checks. Inspecting architecture before "
        "choosing a minimal dose-sensitivity edit; factual NLL tolerance is "
        "+0.005 nats per reading.\n"
    )
```