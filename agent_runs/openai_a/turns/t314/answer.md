The architecture confirms that the current edit changes the reading head, not the recurrent dynamics. The notebook supports an insulin effect much larger than the original rollout predicted; norepinephrine remains uncertain. I’ll inspect the saved helpers in bounded output so the next trial is reproducible and reversible.

```python
import inspect, types
from pathlib import Path

fns = sorted(k for k, v in list(globals().items())
             if isinstance(v, types.FunctionType) and not k.startswith("_"))
print("FUNCTIONS:", fns)

print("\nEDIT/CACHE OBJECT NAMES:")
selected = []
for k, v in sorted(list(globals().items())):
    if any(s in k.lower() for s in
           ("blend", "graft", "checkpoint", "cache", "activity", "separator")):
        if isinstance(v, dict):
            desc = "dict keys=" + str(list(v)[:8])
        elif hasattr(v, "shape"):
            desc = type(v).__name__ + " " + str(tuple(v.shape))
        elif isinstance(v, types.FunctionType):
            desc = str(inspect.signature(v))
        else:
            desc = type(v).__name__ + " " + repr(v)[:90]
        selected.append(k + ": " + desc)
print("\n".join(selected[:65]))

print("\nNOTEBOOK SECTION TITLES:")
titles = [s for s in Path("/work/notes.md").read_text().splitlines()
          if s.startswith("##")]
print("\n".join(titles[:55]))
```
