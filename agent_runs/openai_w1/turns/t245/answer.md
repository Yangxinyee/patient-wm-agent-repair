The architecture confirms that dose signals reach future predictions through the recurrent state, while previous readings also enter the value head directly. The recovered regressions suggest opposite drug effects, but do not by themselves rule out confounding. The output was truncated, so I’ll retrieve only the strongest identification evidence and the latest relay variables. Those must support both the effect targets and a factual-safe transmission path.

```python
import inspect
from pathlib import Path

notes217 = Path("/work/notes.md").read_text()
pars217 = notes217.split("\n\n")

print("=== EFFECT-IDENTIFICATION SUMMARIES ===")
picked217 = [
    p for p in pars217
    if any(s in p.lower() for s in
           ["six-lag", "impulse", "distributed lag", "lag kernel",
            "negative control", "sequential", "placebo"])
]
for p in picked217[:12]:
    print(p[:700])

print("\n=== NUMERIC-AMPLIFICATION RESULTS ===")
picked217 = [
    p for p in pars217
    if any(s in p.lower() for s in ["5x", "numeric amplification", "numeric-edit"])
]
for p in picked217[:10]:
    print(p[:650])

print("\n=== LATEST IMPLEMENTATION OBJECTS ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    relevant = (
        name.endswith(("213", "214", "215"))
        or any(s in name.lower() for s in ["relay", "marker", "carrier"])
    )
    if relevant:
        if callable(obj):
            try:
                desc = str(inspect.signature(obj))
            except Exception:
                desc = type(obj).__name__
        elif hasattr(obj, "shape"):
            desc = str(tuple(obj.shape))
        elif isinstance(obj, dict):
            desc = "keys=" + str(list(obj)[:8])
        elif isinstance(obj, (list, tuple)):
            desc = "len=" + str(len(obj))
        else:
            desc = type(obj).__name__
        print(name, desc)

print("\n=== CPU / ATTENTION FUNCTION NAMES ===")
for name, obj in list(globals().items()):
    if callable(obj) and not name.startswith("_") and any(
        s in name.lower() for s in ["core", "cpu", "attn", "branch", "prelude"]
    ):
        try:
            print(name, inspect.signature(obj))
        except Exception:
            print(name, type(obj).__name__)

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 217\n"
            "Architecture recovery complete; prior output truncated before "
            "implementation inventory. Retrieving compact identification, "
            "numeric-amplification results, and latest marker/relay objects. "
            "No weights changed. Counts: 217 Python runs, 20 forwards.\n")
```