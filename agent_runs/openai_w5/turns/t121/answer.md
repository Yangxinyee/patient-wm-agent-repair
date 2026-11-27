The earlier record analyses suggest insulin lowers glucose and norepinephrine raises MAP, while the original rollouts respond only weakly. Those estimates still depend on confounding and persistence assumptions. I’ll recover the saved diagnostics and channel-building helpers; the next edit needs both reliable dose encoding and factual centering, not merely a large fitted response.

```python
import inspect
from pathlib import Path

notes = Path("/work/notes.md").read_text()
print("=== EFFECT-ESTIMATION NOTES ===")
print(notes[6500:11200])

print("\n=== FUNCTION NAMES ===")
print(" ".join(sorted(
    n for n, x in list(globals().items())
    if not n.startswith("_") and inspect.isfunction(x)
)))

print("\n=== RELEVANT OBJECTS ===")
for n, x in sorted(list(globals().items())):
    if n.startswith("_") or not any(s in n.lower() for s in
        ("track", "temp", "rate", "channel", "clear", "panel", "candidate",
         "fit105", "ridge", "state", "dose", "coef", "direction", "tune", "feature")):
        continue
    if inspect.isfunction(x):
        print(n, str(inspect.signature(x)))
    elif isinstance(x, dict):
        print(n, "dict", len(x), "keys", list(x)[:12])
    elif hasattr(x, "shape") and not callable(getattr(x, "shape", None)):
        print(n, type(x).__name__, tuple(x.shape))
    elif isinstance(x, (list, tuple)):
        print(n, type(x).__name__, len(x), repr(x)[:220])
    elif isinstance(x, (str, int, float, bool)):
        print(n, repr(x)[:220])

print("\n=== CHANNEL-RELEVANT TENSORS ===")
for n in NAMES:
    if any(s in n for s in (
        "encode_action", "core.flow", "core.c_out", "core.coda",
        "core.prelude", "attr_heads.value_bin", "last_numeric",
        "core.out_norm", "query_norm", "embed.numeric"
    )):
        print(n, SHAPES[n])
```
