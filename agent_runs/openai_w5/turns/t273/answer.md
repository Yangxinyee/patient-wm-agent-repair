The architecture confirms that doses reach predictions through the recurrent state, while generated readings are fed back during rollouts. The notebook also records nonzero effects estimated from the records, but the output was truncated before the edit details. I’ll retrieve those details selectively. If centering responds to edited doses, it could cancel part of the intended correction; otherwise I’ll test head strength with paired factual checks.

```python
import os, json, numpy as np, pandas as pd, torch
from pathlib import Path

notes_text = Path("/work/notes.md").read_text()
lines = notes_text.splitlines()
print("NOTEBOOK CHARACTERS", len(notes_text))
print("HEADINGS")
print("\n".join(x for x in lines if x.startswith("## ")))
print("\nIDENTIFICATION AND TRANSPORT NOTES")
terms = ("identif", "random", "confound", "deconvol", "placebo",
         "negative control", "clean pair", "centering", "coordinates", "reserved")
selected = [(i, x) for i, x in enumerate(lines)
            if any(t in x.lower() for t in terms)]
for i, x in selected[-45:]:
    print(i, x[:450])

print("\nKERNEL NAMES")
ks = [k for k in globals() if not k.startswith("_")]
print(" ".join(ks))
print("\nDICT/FUNCTION SUMMARIES")
for k in ks:
    v = globals().get(k)
    if isinstance(v, dict) and k not in ("VOCAB", "SHAPES", "__builtins__"):
        print(k, "dict", len(v), list(v)[:10])
    elif callable(v) and k not in ("load", "load_original", "save_edit", "drop_edit", "edited"):
        print(k, "callable")
print("\nFORWARD 22 FILES", os.listdir("/work/fwd_22"))
print("\nVOCAB CONFIG", VOCAB.get("config"))
print("CODES", VOCAB.get("codes"))
```
