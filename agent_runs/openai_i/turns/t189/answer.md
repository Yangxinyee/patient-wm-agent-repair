The rollout rejects this candidate: insulin barely changed, and doubled norepinephrine lost its positive response. Passing factual fit was not enough. I’ll revert the probe and inspect the architecture and retained calibration objects. The next edit needs a directly verified downstream response, not just dose-sensitive features.

```python
import numpy as np, inspect
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

candidate_probe170 = {}
for name in list(edited()):
    value = load(name)
    candidate_probe170[name] = value.clone() if hasattr(value, "clone") else value.copy()
    drop_edit(name)
print("\nActive edits after reversion:", edited())

print("\n=== RETAINED OBJECTS ===")
inventory170 = []
for key, value in list(globals().items()):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB", "inventory170", "value"}:
        continue
    if inspect.ismodule(value):
        continue
    if callable(value):
        try:
            detail = str(inspect.signature(value))
        except Exception:
            detail = ""
    elif hasattr(value, "shape") and not callable(value.shape):
        detail = str(tuple(value.shape))
    elif isinstance(value, dict):
        detail = "keys=" + repr(list(value.keys())[:12])
    elif isinstance(value, (list, tuple)):
        detail = "len=" + str(len(value))
    elif isinstance(value, (str, int, float, bool)):
        detail = repr(value)[:100]
    else:
        detail = ""
    inventory170.append((key, type(value).__name__, detail))
for row in sorted(inventory170):
    print(row)

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 170; forwards 11–12 — reject response-based posterior probe\n"
            "- On 50 tuning stays, factual NLL changed +0.00025 (passes this local safeguard).\n"
            "- Matched rollouts: insulin x2 -2.148 -> -2.112; half .819 -> .797; "
            "NE x2 .471 -> -.007; half -.115 -> .006.\n"
            "- Posterior feature sensitivity did not produce the intended downstream effects. "
            "Reverted all four edits and saved candidate_probe170.\n"
            "- Next: inspect architecture and directly calibrate a controllable downstream pathway.\n")
```