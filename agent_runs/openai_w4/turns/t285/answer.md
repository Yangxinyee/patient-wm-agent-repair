The architecture confirms that an edit can affect both immediate predictions and later generated readings, so factual fit alone is insufficient. The inspection output was truncated before the useful repair variables. I’ll retrieve those compactly so the next test changes only insulin’s dose-innovation gain and gate, leaving the compensating edits intact.

```python
import re, types
_s248 = list(globals().items())

_funcs248 = [n for n, o in _s248
             if isinstance(o, types.FunctionType) and not n.startswith("_")]
print("FUNCTION NAMES:", ", ".join(_funcs248))

_keys248 = [n for n, o in _s248 if not n.startswith("_")
            and any(t in n.lower() for t in
                    ("r244", "r245", "r246", "backup", "kernel", "target",
                     "gate", "innovation", "carrier", "policy"))]
print("RELEVANT VARIABLE NAMES:", ", ".join(_keys248))

print("LATE CANDIDATE OBJECTS:")
for n, o in _s248:
    if n.startswith(("r244", "r245", "r246")):
        if isinstance(o, types.FunctionType):
            c = o.__code__
            print(n, "args", c.co_varnames[:c.co_argcount],
                  "globals", c.co_names)
        elif isinstance(o, dict):
            print(n, "dict", [(str(k)[:65],
                               tuple(v.shape) if hasattr(v, "shape")
                               else str(v)[:160]) for k, v in list(o.items())[:12]])
        elif hasattr(o, "shape"):
            print(n, "shape", tuple(o.shape))
        else:
            print(n, repr(o)[:250])

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 247–248\n"
            "Read architecture and record schema. Run-247 display truncated before "
            "candidate metadata; Run-248 requests bounded construction metadata. "
            "No saved weights changed. Gate-40 candidate remains GPU factual-safe "
            "on the tested subset but under-responsive for insulin.\n")
```
