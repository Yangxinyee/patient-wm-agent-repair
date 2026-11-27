The stronger direct-token edit preserved factual fit but barely changed the dose responses, so increasing its gain is not useful. The attention-routed edit responded more strongly but failed the factual limit. I’ll recover the architecture and calibration evidence before testing a targeted compensation. It would succeed only if it restores factual fit without suppressing the dose effects.

```python
from pathlib import Path
import re
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== READOUT TENSORS ===")
for name in NAMES:
    if re.search(r"head|readout|decode|value|fine|coarse", name, re.I):
        print(name, SHAPES[name])

notes292 = Path("/work/notes.md").read_text()
lines292 = notes292.splitlines()
matches292 = [
    (i, line) for i, line in enumerate(lines292)
    if re.search(r"record estimates|causal targets|target shifts|true effects|per rate unit|IV estimate|instrument", line, re.I)
]
print("\n=== RECORD-BASED EFFECT EVIDENCE ===")
selected292 = matches292[:8] + matches292[-8:]
seen292 = set()
for i, line in selected292:
    if i not in seen292:
        seen292.add(i)
        print("\n".join(lines292[max(0, i-1):min(len(lines292), i+3)]))

print("\n=== AVAILABLE CALIBRATION OBJECTS ===")
items292 = list(globals().items())
for name, obj in items292:
    if re.search(r"cal|target|readout|logit|compens|feature|head|resid", name, re.I):
        if isinstance(obj, (np.ndarray,)):
            print(name, obj.shape, obj.dtype)
        elif hasattr(obj, "shape") and hasattr(obj, "dtype"):
            print(name, tuple(obj.shape), obj.dtype)
        elif callable(obj) and hasattr(obj, "__code__"):
            print(name, "function", obj.__code__.co_varnames[:obj.__code__.co_argcount])
        elif isinstance(obj, dict):
            print(name, "dict", len(obj), "keys:", list(obj)[:5])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 292 — recover targeted compensation options\n"
        "- Higher direct-token gains were factually acceptable but causally ineffective; the low-gain checkpoint remains installed.\n"
        "- Recovering architecture, record-based effect evidence, and available readout/calibration features before spending the remaining runs.\n"
        "- No weights changed. Any routed candidate must pass independent factual validation and retain its dose responses.\n"
    )
```