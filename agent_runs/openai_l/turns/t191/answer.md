The quiet dose write barely changed factual fit, but the latest contrasts still show weak own-drug responses and unwanted cross-drug responses. I’ll recover the decoder setup and record-based effect estimates before fitting a correction. The hypothesis is that a selective readout can recover the supported effects without moving factual predictions; stronger dose contrasts with unchanged factual loss would support it.
```python
from pathlib import Path
import inspect, re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

notes174 = Path("/work/notes.md").read_text()
print("\n=== EARLY CAUSAL EVIDENCE / BASELINES ===")
print(notes174[:14500])

print("\n=== AVAILABLE FUNCTIONS ===")
for name174, obj174 in sorted(list(globals().items())):
    if inspect.isfunction(obj174):
        try:
            sig174 = str(inspect.signature(obj174))
        except Exception:
            sig174 = "?"
        print(name174, sig174)

print("\n=== RECENT VARIABLES ===")
for name174, obj174 in sorted(list(globals().items())):
    if re.search(r"(16[0-9]|17[0-3])$", name174):
        if isinstance(obj174, dict):
            print(name174, "dict", [(str(k), getattr(v, "shape", type(v).__name__))
                                     for k, v in list(obj174.items())[:16]])
        elif hasattr(obj174, "shape"):
            print(name174, tuple(obj174.shape))
        elif isinstance(obj174, (list, tuple)):
            print(name174, type(obj174).__name__, "length", len(obj174),
                  "first", repr(obj174[:5])[:300])
        else:
            print(name174, type(obj174).__name__, repr(obj174)[:200])

with open("/work/notes.md", "a") as f174:
    f174.write("\n## Run 174 / decoder recovery\n"
               "Quiet dose write remains factual-neutral on the prior audit (+~0.00006 NLL). "
               "Recovered architecture, empirical-effect evidence, and live helper interfaces "
               "before fitting a factual-anchored value readout. No new edits.\n")
```