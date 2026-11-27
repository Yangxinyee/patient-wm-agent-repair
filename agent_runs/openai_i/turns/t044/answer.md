The posterior reconstruction now matches the exposed latent means almost exactly. That gives us a reliable place to test dose-sensitive edits, but does not establish their causal accuracy. I’ll recover the architecture and earlier identification evidence before choosing an edit; a useful correction must generalize beyond the records used to fit it.

```python
from pathlib import Path
import numpy as np
import pandas as pd

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK ===")
notes_text = Path("/work/notes.md").read_text()
print(notes_text[:6500])

print("\n=== RETAINED OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or name in {"NAMES", "SHAPES", "VOCAB", "notes_text"}:
        continue
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print(name, "function")
    elif isinstance(obj, dict):
        keys = list(obj)
        print(name, "dict", len(keys), "keys:", keys[:12])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 37; forward 6 — restart recovery\n"
            "- No weights changed. Posterior reconstruction matches exposed means to about 4e-6 RMS.\n"
            "- Recovering architecture and record-based identification before designing a posterior edit.\n")
```