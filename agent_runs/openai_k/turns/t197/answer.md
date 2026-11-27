The output-layer correction barely changed dose sensitivity at the low-distortion setting; stronger settings mainly disturbed factual predictions. I’ll recover the exact model path and retained effect estimates before saving it. A useful edit must change the paired dose response, not just the baseline predictions.

```python
from pathlib import Path
import inspect, re, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
notes181 = Path("/work/notes.md").read_text()
print("\n=== EARLY IDENTIFICATION AND SAVED-EDIT NOTES ===")
sections181 = re.split(r"(?=^## )", notes181, flags=re.M)
for sec in sections181:
    heading = sec.splitlines()[0] if sec.splitlines() else ""
    if re.search(r"\b(1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9]|10[0-9]|11[0-9]|12[0-9]|13[0-9]|140)\b", heading):
        if any(x in sec.lower() for x in ["identif", "candidate139", "persistence", "true", "confound", "instrument", "random"]):
            print(sec[:2600])
print("\n=== RECENT RETAINED INTERFACES ===")
for name181 in sorted(list(globals())):
    if re.search(r"(139|151|157|158|166|179|180)$", name181):
        obj181 = globals()[name181]
        if callable(obj181):
            try:
                print(name181, "function", inspect.signature(obj181))
            except Exception:
                print(name181, "callable")
        elif isinstance(obj181, dict):
            print(name181, "dict keys", list(obj181)[:35])
        elif isinstance(obj181, (list, tuple)):
            print(name181, type(obj181).__name__, "length", len(obj181))
        else:
            print(name181, type(obj181).__name__, "shape", getattr(obj181, "shape", None))
print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f181:
    f181.write("\n## Run 181: restart and exact-path audit\n"
               "- Run 180's low-distortion head correction barely changed drug sensitivity; it remains unsaved.\n"
               "- Recovering architecture, identification evidence, saved candidate139, and recent interfaces before another edit.\n"
               "- Python runs: 181; forward passes: 10. No weights changed.\n")
```