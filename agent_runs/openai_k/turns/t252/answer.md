The architecture confirms that every head reads a normalized recurrent state, so a large latent injection need not produce a large reading change. The output was cut off before the diagnostic definitions. I’ll retrieve only those definitions and their inputs now. If the problem is an argument mismatch, I can fix it; if replay still disagrees with official states, I’ll reject it as an editing guide.

```python
import inspect, dis, io
def brief226(x):
    if hasattr(x, "shape"):
        return (type(x).__name__, tuple(x.shape))
    if isinstance(x, dict):
        return {str(k): brief226(v) for k, v in list(x.items())[:18]}
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, [brief226(v) for v in x[:8]])
    if isinstance(x, (int, float, str, bool, type(None))):
        return repr(x)[:150]
    return type(x).__name__

print("=== CORE HELPER ===")
f226 = globals().get("repaired_core_local")
if callable(f226):
    print("args:", f226.__code__.co_varnames[:f226.__code__.co_argcount])
    print("defaults:", brief226(f226.__defaults__))
    try:
        text226 = inspect.getsource(f226)
    except (OSError, TypeError):
        b226 = io.StringIO()
        dis.dis(f226, file=b226)
        text226 = b226.getvalue()
    print(text226[:5200])

print("=== TOKEN HELPER ===")
f226 = globals().get("local_token154")
if callable(f226):
    print("args:", f226.__code__.co_varnames[:f226.__code__.co_argcount])
    print("defaults:", brief226(f226.__defaults__))
    print("names:", f226.__code__.co_names)
    print("constants:", [x for x in f226.__code__.co_consts
                         if isinstance(x, (str, int, float, tuple))][:25])

print("=== AVAILABLE OBJECTS ===")
for key226 in ("context151", "candidate158", "backup218", "W0",
               "w", "weights", "state_keys_audit", "groups59", "r49_readouts"):
    if key226 in globals():
        print(key226, brief226(globals()[key226]))
print("source/history objects:",
      [k for k in list(globals())
       if any(s in k.lower() for s in ("source", "script", "history", "code154"))][:35])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 226: focused replay definitions\n"
            "- Architecture recovered; prior output truncation hid diagnostics.\n"
            "- Inspecting the core helper and argument types only, to resolve tuple indexing.\n"
            "- No weights changed. Python runs: 226; forward passes: 17.\n")
```