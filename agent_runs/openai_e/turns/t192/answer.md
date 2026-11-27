The last audit supports the alignment of readings with prior model states, but one bin-boundary convention still differs. I’ll recover the decoder and earlier effect estimates before fitting a small factual calibration. The hypothesis is that calibration can recover NLL margin without weakening dose responses; paired factual and rollout tests will confirm or refute it.

```python
from pathlib import Path
import re, numpy as np

arch163 = Path("/model/ARCHITECTURE.md").read_text()
print("ARCHITECTURE", len(arch163), "characters")
print(arch163[:18000])
print("\nDATA DEFINITIONS")
print(Path("/data/README.md").read_text()[:5000])

notes163 = Path("/work/notes.md").read_text()
print("\nEARLY EVIDENCE")
print(notes163[:8500])
sections163 = re.split(r"\n(?=## )", notes163)
selected163 = [
    s for s in sections163
    if re.search(r"^## .*?(?:Run 77\b|Run 103\b|Forward 11\b|Forward 12\b|Run 159\b|Run 160\b|Run 161\b|Run 162\b)", s)
]
print("\nSELECTED VALIDATION / CALIBRATION NOTES")
for s in selected163:
    print(s[:2200])

print("\nLIVE NOTEBOOK FUNCTIONS")
for k, v in list(globals().items()):
    if callable(v) and getattr(v, "__module__", "") == "__main__" and not k.startswith("_"):
        print(k)

print("\nRELEVANT LIVE OBJECTS")
for k, v in list(globals().items()):
    if not k.startswith("_") and re.search(
        r"(decoder|dec_|reading|meta|target|effect|calib|baseline|static|train|reg)", k, re.I
    ):
        if hasattr(v, "shape"):
            print(k, type(v).__name__, tuple(v.shape))
        elif isinstance(v, dict):
            print(k, "dict", len(v), "keys:", list(v)[:12])
        elif isinstance(v, (list, tuple)):
            print(k, type(v).__name__, len(v))
        elif callable(v):
            pass
        elif isinstance(v, (int, float, str)):
            print(k, repr(v)[:200])

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 163: restart recovery before factual calibration\n"
        "Read architecture first, recovered early record evidence and selected validation notes, "
        "and inventoried decoder/calibration objects. No weights changed. "
        "Register alignment is supported; bin-boundary convention still needs resolution. "
        "Python runs: 163; forwards: 22.\n"
    )
```