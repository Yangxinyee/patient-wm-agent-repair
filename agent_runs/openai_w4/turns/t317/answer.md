The norepinephrine candidates stayed within the CPU factual-fit budget, but that does not tell us whether they improve dose rollouts. I’ll recover their exact construction before choosing one for GPU validation. It will be promising only if both norepinephrine readouts move toward the record-based estimates while factual NLL remains within +0.005.

```python
from pathlib import Path
import inspect

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nRELEVANT GLOBALS")
keys274 = sorted(
    k for k in list(globals())
    if not k.startswith("_")
    and (k.startswith("ne") or any(s in k for s in ("271", "272", "273"))
         or k.startswith("make_"))
)
for k in keys274:
    v = globals()[k]
    if callable(v):
        c = getattr(v, "__code__", None)
        print(k, "CALLABLE", inspect.signature(v),
              "names:", c.co_names if c else "")
        try:
            print(inspect.getsource(v))
        except (OSError, TypeError):
            pass
    elif isinstance(v, dict):
        print(k, "DICT", len(v), "keys:", list(v)[:25])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    else:
        print(k, type(v).__name__, repr(v)[:350])

print("\nCURRENT EDITS", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 274\n"
            "Recovered architecture and recent NE candidate construction metadata. "
            "Run-273 CPU screening alone does not establish rollout improvement. "
            "No submitted weights changed.\n")
```