The latent state has a relatively clear norepinephrine-dose signal, but we have not established how it influences MAP predictions. I’ll recover the architecture and saved estimators before choosing a selective latent edit. Success would mean stronger dose responses while keeping factual NLL within the allowed increase.

```python
from pathlib import Path
import numpy as np

print(Path("/model/ARCHITECTURE.md").read_text())
print("\nSAVED ANALYSIS VARIABLES")
for key in sorted(list(globals())):
    if key.startswith("_") or key in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    value = globals()[key]
    if isinstance(value, np.ndarray):
        print(key, "array", value.shape)
    elif isinstance(value, dict):
        print(key, "dict keys:", list(value)[:14])
    elif isinstance(value, (list, tuple)) and len(value) < 30:
        print(key, type(value).__name__, "length", len(value))
print("\nCURRENT OPERATIONS", ops())
with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 37: restart inspection\n"
            "- Numeric gain 4 remains installed. Recovering architecture and saved estimators before designing a latent-state correction.\n"
            "- Last diagnostic shows a strong NE signal in z but does not establish that amplifying it produces the correct MAP response.\n"
            "- Forward passes used: 19.\n")
```