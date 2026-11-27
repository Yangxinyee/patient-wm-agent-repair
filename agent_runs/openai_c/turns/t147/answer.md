Selective recurrent amplification still left insulin responses far below the record-based effects. I’ll remove it and review how earlier head edits handled the last-reading register. I’m checking whether those edits mostly disturbed reading history rather than amplifying treatment information; better separation would justify another head test.

```python
from pathlib import Path
import re
import pandas as pd
import numpy as np

undo()
undo()
print("RESTORED", ops())

sections = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
wanted = (
    "Python run 53", "Forward 23", "Python run 59",
    "Python run 71", "Python run 72", "Python run 73",
    "Forward 28", "Python run 76", "Python run 87"
)
for section in sections:
    heading = section.splitlines()[0] if section.splitlines() else ""
    if any(heading.startswith("## " + prefix) for prefix in wanted):
        print(section[:1000])

print("\nSAVED DATAFRAME SCHEMAS")
for key, value in sorted(list(globals().items())):
    if isinstance(value, pd.DataFrame) and any(
        term in key.lower() for term in
        ("state", "row", "read", "trans", "action", "join", "design", "sensor", "72", "71", "59")
    ):
        print(key, value.shape, list(value.columns))

print("\nHEAD AND REGISTER TENSORS")
for name in NAMES:
    if any(term in name for term in
           ("attr_heads.value", "last_numeric", "query_norm",
            "embed_transition.value")):
        print(name, SHAPES[name])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 104 / forward 40: selective recurrent pilot rejected\n"
            "- Recurrent two-direction gain 4: insulin -2.409/+1.568; NE +0.171/-0.708.\n"
            "- No material main-readout improvement; removed both recurrent operations.\n"
            "- Numeric gain 4 restored; auditing head/register separation before another candidate.\n"
            "- Forward passes used: 40; 10 remain.\n")
```