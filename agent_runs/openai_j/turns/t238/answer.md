The architecture confirms that the current edit is confined to the value head; it does not change the recurrent state or other heads during teacher-forced prediction. The inspection output was truncated before the candidate details appeared. I’ll retrieve those details next, looking for whether doubled and halved norepinephrine produce distinct, usable feature changes.

```python
import re
import inspect
from pathlib import Path

print("=== RECENT CANDIDATE VARIABLES ===")
selected = [
    n for n in sorted(list(globals()))
    if re.search(r"(177|192|19[7-9]|20[0-9]|210)", n)
    and not n.startswith("_")
]
for n in selected:
    obj = globals()[n]
    if callable(obj):
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = ""
        code = getattr(obj, "__code__", None)
        print(n, sig)
        if code is not None:
            print("  variables:", code.co_varnames)
            print("  references:", code.co_names)
    elif isinstance(obj, dict):
        print(n, "dict", list(obj)[:22])
    elif hasattr(obj, "shape"):
        print(n, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, (list, tuple)):
        print(n, type(obj).__name__, "length", len(obj))
    else:
        print(n, type(obj).__name__, repr(obj)[:120])

print("\n=== CACHED EVALUATOR AND CONTRAST NAMES ===")
for n in sorted(list(globals())):
    if any(s in n.lower() for s in ("tilt", "contrast", "probe", "wm_train", "wm_tun")):
        obj = globals()[n]
        if isinstance(obj, dict):
            print(n, "keys", list(obj)[:18])
        elif callable(obj):
            try:
                print(n, inspect.signature(obj))
            except Exception:
                print(n, "callable")
        elif hasattr(obj, "shape"):
            print(n, tuple(obj.shape))

print("\n=== IDENTIFICATION SUMMARY ===")
text = Path("/work/notes.md").read_text()
start = text.find("## Causal identification checks")
print(text[start:start + 4300])
```