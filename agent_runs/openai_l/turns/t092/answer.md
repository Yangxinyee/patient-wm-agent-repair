The head-only correction shows a trade-off: stronger edits improve dose sensitivity but worsen factual predictions. The safer settings remain too weak. I’ll recover the fitting code and model interfaces to find a more selective edit. Success requires stronger dose responses with factual loss below the 0.005-nat limit; exceeding that limit would reject the edit.

```python
from pathlib import Path
import re

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== RECENT KERNEL OBJECTS ===")
for name in sorted(list(globals())):
    if re.search(r"(7[7-9]|8[0-3])$", name):
        obj = globals()[name]
        shape = getattr(obj, "shape", None)
        if shape is not None:
            desc = f"shape={tuple(shape)}"
        elif isinstance(obj, dict):
            desc = f"keys={list(obj)[:18]}"
        elif isinstance(obj, (list, tuple)):
            desc = f"len={len(obj)}"
        elif callable(obj):
            desc = "callable"
        else:
            desc = repr(obj)[:180]
        print(name, type(obj).__name__, desc)

print("\n=== RETAINED FUNCTIONS ===")
print([n for n, x in list(globals().items())
       if callable(x) and not n.startswith("_")
       and (re.search(r"\d\d$", n) or
            any(s in n.lower() for s in ("encode", "transition", "decode", "kernel", "effect", "nll")))])
print("\n=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 84 / restart recovery\n"
            "Run 83's head-only ridge edit cannot reach the causal slopes without factual loss: "
            "lambda=1 held-out dNLL=.001114, slopes I~-0.89/NE~.43-.58; "
            "lambda=.3 held-out dNLL=.006454 exceeds the .005 limit. "
            "Recovering fit interfaces before choosing a more selective edit. No tensors edited.\n")
```