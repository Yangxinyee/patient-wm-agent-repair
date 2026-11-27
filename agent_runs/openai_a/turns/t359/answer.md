The pooled rollouts show stronger insulin effects and almost unchanged norepinephrine effects. The three omitted stays have no post-hour-12 target readings, explaining their absence. Factual NLL remains within the allowed increase. I’ll audit the record-based evidence and saved weights without further tuning; contradictory evidence or a checkpoint mismatch would challenge this choice.

```python
from pathlib import Path
import re
import numpy as np

_arch295 = Path("/model/ARCHITECTURE.md").read_text()
_notes295 = Path("/work/notes.md").read_text()
_sections295 = re.split(r"(?=^## )", _notes295, flags=re.MULTILINE)

print("=== ARCHITECTURE ===")
print(_arch295[:16000])
print("\n=== NOTEBOOK EVIDENCE INDEX ===")
for i, section in enumerate(_sections295):
    title = section.splitlines()[0] if section.strip() else ""
    if re.search(
        r"identif|causal|instrument|insulin|target|dynamic|graft|gate|amplitude|final|safety",
        title, re.IGNORECASE
    ):
        print(f"{i}: {title}")

print("\n=== CURRENT EDITS ===")
print(edited())
print("\n=== CHECKPOINT INTEGRITY ===")
_ck295 = globals().get("amplitude093_checkpoint")
if isinstance(_ck295, dict):
    for name in edited():
        if name in _ck295:
            current = np.asarray(load(name))
            reference = np.asarray(_ck295[name])
            print(name, "shape", current.shape,
                  "exact_match", np.array_equal(current, reference),
                  "max_abs_difference", float(np.max(np.abs(current - reference))))
        else:
            print(name, "not present in checkpoint dictionary")
else:
    print("Checkpoint structure:", type(_ck295).__name__)
```
