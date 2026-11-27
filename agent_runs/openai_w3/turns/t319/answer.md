The last replay showed effects with mostly correct signs, but weak immediate responses and leakage into unrelated sensors. My hypothesis is that generated-history feedback, rather than simple dose scaling, causes the rollout errors. I’ll recover the transition code and record-derived dynamics; accumulated responses that diverge from those dynamics would support this hypothesis, while accurate accumulation would refute it.

```python
from pathlib import Path
import inspect, re, numpy as np, torch

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nRECORD SCHEMA")
print(Path("/data/README.md").read_text())

print("\nREPLAY FUNCTIONS")
for name in ["preludeE185", "prelude80", "state_context", "context_with_flow",
             "encode_groups80", "encode_bundles", "view171", "phys171",
             "readout134", "tensor153", "mlp", "linear"]:
    obj = globals().get(name)
    if callable(obj):
        try:
            text = inspect.getsource(obj)
            print(name, text[:7000])
        except Exception:
            code = getattr(obj, "__code__", None)
            print(name, inspect.signature(obj))
            if code:
                print("names:", code.co_names, "constants:", repr(code.co_consts)[:2200])

print("\nRELEVANT CACHE INVENTORY")
pat = re.compile(r"checkpoint|replay|transition|prelude|readout|raw|direction|effect|target|response|pairs|calib|original|phys|source|script|^S$|^W$|^rs28|^ss28|^codes")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_") or not pat.search(name):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys", list(obj)[:12])
    elif isinstance(obj, str):
        print(name, "str", len(obj), repr(obj[:160]))
    elif callable(obj):
        try:
            print(name, "function", inspect.signature(obj))
        except Exception:
            pass
    elif isinstance(obj, (list, tuple)):
        print(name, type(obj).__name__, len(obj), repr(obj[:3])[:260])
    elif isinstance(obj, (int, float, np.number)):
        print(name, obj)

print("\nIDENTIFICATION / DYNAMICS NOTEBOOK EXCERPTS")
lines = Path("/work/notes.md").read_text().splitlines()
hits = [i for i, s in enumerate(lines)
        if re.search(r"identif|autoregress|instrument|randomiz|residual.*dose|true effect|effect.*estimat|persistence|rho|lag coefficient", s, re.I)]
for i in hits[:18]:
    print("\n".join(lines[max(0, i-1): min(len(lines), i+3)])[:1800])

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 281; forwards 24. Latest one-step response is weak and cross-contaminated despite oversized insulin rollout effect. Recovering exact transition and record-only identification evidence before testing accumulated feedback. No weights changed.\n")
```
