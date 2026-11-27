The large-gain probe preserved factual fit but did not repair the dose responses, so that latent direction is rejected and all weights are original again. Next I’ll recover the exact state-update equations and record-derived targets. A useful replacement direction must change future predictions in a measured forward calculation—not merely correlate with recorded readings.

```python
from pathlib import Path
import re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

notes133 = Path("/work/notes.md").read_text()
print("\n=== EARLIER IDENTIFICATION EVIDENCE ===")
sections133 = re.split(r"(?=^## )", notes133, flags=re.M)
selected133 = [
    s for s in sections133
    if any(w in s.lower() for w in
           ["impulse", "causal", "instrument", "identification",
            "cumulative", "record-derived", "confound"])
]
for s in selected133[:18]:
    print(s[:1800])

print("\n=== POSSIBLE CORE / RESPONSE HELPERS ===")
for name133, obj133 in list(globals().items()):
    if callable(obj133) and any(w in name133.lower() for w in
                               ["core", "attention", "block", "response", "step", "repair"]):
        code133 = getattr(obj133, "__code__", None)
        if code133 is not None:
            print(name133, "args:", code133.co_varnames[:code133.co_argcount],
                  "uses:", code133.co_names[:35])

print("\n=== SAVED CANDIDATE STRUCTURE ===")
for name133 in ["candidate_small131", "candidate_large132"]:
    obj133 = globals().get(name133)
    print(name133, type(obj133).__name__)
    if isinstance(obj133, dict):
        print([(k, getattr(v, "shape", type(v).__name__)) for k, v in obj133.items()])

with open("/work/notes.md", "a") as f133:
    f133.write(
        "\n## Python 133 — forward-response calibration restart\n"
        "- All edits remain reverted after the failed large-gain probe.\n"
        "- Recovering exact recurrent equations and existing record-derived targets.\n"
        "- New directions must be selected by finite-difference future prediction responses, "
        "not posterior/reading correlations.\n"
    )
```