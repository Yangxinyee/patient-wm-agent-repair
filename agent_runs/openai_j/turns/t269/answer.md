The architecture confirms that our edit changes only the value head; rollout feedback can still affect other predictions. The output cut off before the causal evidence and helper names. I’ll recover those next, keeping the validated edit unchanged unless a new correction improves held-out doubling while preserving factual fit.

```python
from pathlib import Path
import inspect
import pandas as pd

_g239 = dict(globals())
_keys239 = [
    k for k in sorted(_g239)
    if not k.startswith("_")
    and any(s in k.lower() for s in
            ["233", "234", "235", "236", "237", "head177", "map192"])
]
_lines239 = []
for k in _keys239:
    v = _g239[k]
    if callable(v):
        try:
            desc = str(inspect.signature(v))
        except Exception:
            desc = "callable"
    elif isinstance(v, dict):
        desc = "dict " + str(len(v)) + " keys " + str(list(v)[:6])
    elif hasattr(v, "shape"):
        desc = str(v.shape)
    else:
        desc = str(v)[:100]
    _lines239.append(k + ": " + desc)

print("KERNEL OBJECTS")
print("\n".join(_lines239)[:3200])

print("\nHEAD CONSTRUCTION")
if "head177_make" in _g239:
    try:
        print(inspect.getsource(_g239["head177_make"])[:1800])
    except Exception:
        fn = _g239["head177_make"]
        print("names:", fn.__code__.co_names)
        print("arguments/local names:", fn.__code__.co_varnames)

_notes239 = Path("/work/notes.md").read_text()
_pos239 = _notes239.find("## Causal identification checks")
print("\nCAUSAL EVIDENCE")
print(_notes239[_pos239:_pos239 + 3000])

print("\nKERNEL TARGETS")
print(pd.read_csv("/work/kernel_targets.csv").head(8).to_string(index=False))
print("\nACTIVE EDITS", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 239: compact recovery\n"
        "- Recovering evidence and helper names after truncated diagnostic output.\n"
        "- No weights changed; gain-8 remains the validated fallback.\n"
    )
```