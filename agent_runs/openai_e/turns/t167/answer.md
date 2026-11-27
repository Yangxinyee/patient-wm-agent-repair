The recovered notes support opposite drug effects from the misleading raw correlations: insulin lowers glucose, while norepinephrine raises MAP. The output was truncated before the candidate inventory, so I’ll retrieve that in a bounded form. This will let me compose a trial correctly without accidentally adding an old correction twice.

```python
import re
import numpy as np
from pathlib import Path

def brief146(x):
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        return {str(k): brief146(v) for k, v in list(x.items())[:8]}
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}[{len(x)}]"
    if callable(x):
        return "callable"
    return repr(x)[:100]

print("REGRESSION OBJECT INVENTORY")
for name146 in sorted(list(globals())):
    if name146.endswith(("143", "144")):
        print(name146, brief146(globals()[name146]))

print("\nSAVED CANDIDATE CONTENTS")
for file146 in ("baseline77.npz", "ne_trial103.npz", "ne_reduced107.npz"):
    with np.load("/work/" + file146, allow_pickle=False) as archive146:
        print(file146, {k: tuple(archive146[k].shape) for k in archive146.files})

text146 = Path("/work/notes.md").read_text()
sections146 = re.split(r"(?=^## )", text146, flags=re.M)
print("\nSELECTED VALIDATION NOTES")
for label146 in ("Run 58:", "Run 63:", "Run 77:", "Run 103:",
                 "Run 107:", "Forward 14:", "Forward 15:"):
    matches146 = [s for s in sections146 if label146 in s.split("\n", 1)[0]]
    for section146 in matches146:
        print(section146[:850])

with open("/work/notes.md", "a") as f146:
    f146.write(
        "\n## Run 146: bounded candidate recovery\n"
        "Retrieved regression object names, saved-candidate tensor layouts, "
        "and selected earlier validation notes. No weights changed. "
        "Python runs: 146; forwards: 15.\n"
    )
```