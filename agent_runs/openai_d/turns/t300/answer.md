The latest screen suggests that cancellation based on the previous reading could reduce the factual-fit penalty. It does not yet show whether rollout effects survive. I’ll recover the candidate-building code and record-derived effect estimates, then test a correction; it succeeds only if factual fit improves without materially weakening the dose responses.

```python
from pathlib import Path
import inspect, re, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

r255_notes = Path("/work/notes.md").read_text()
r255_sections = re.split(r"(?=^## )", r255_notes, flags=re.M)
print("\n=== RECORD-DERIVED CAUSAL FINDINGS ===")
r255_selected = [
    s for s in r255_sections
    if re.search(r"causal|identif|true effect|causal target|instrument|discontinu", s.split("\n", 1)[0], re.I)
]
for s in r255_selected[-6:]:
    print(s[:2200])

def r255_describe(v):
    if callable(v):
        try:
            sig = str(inspect.signature(v))
        except Exception:
            sig = "?"
        names = getattr(getattr(v, "__code__", None), "co_names", ())
        return f"callable {sig}; names={names}"
    if isinstance(v, dict):
        return "dict " + str({
            k: (tuple(x.shape) if hasattr(x, "shape") else
                list(x.keys())[:12] if isinstance(x, dict) else
                x if np.isscalar(x) else type(x).__name__)
            for k, x in list(v.items())[:18]
        })
    if hasattr(v, "shape"):
        return f"{type(v).__name__} shape={tuple(v.shape)}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__} len={len(v)}; first={str(v[:4])[:180]}"
    return str(v)[:240]

print("\n=== LAST-BIN CANCELLATION INTERFACES ===")
r255_globals = dict(globals())
for n in sorted(r255_globals):
    if n.startswith(("r252_", "r254_")):
        print(n, r255_describe(r255_globals[n]))

print("\n=== CANDIDATE AND CHECKPOINT INTERFACES ===")
for n in sorted(r255_globals):
    if n.startswith(("r244_", "r246_", "r251_")) and (
        callable(r255_globals[n]) or isinstance(r255_globals[n], dict)
    ):
        print(n, r255_describe(r255_globals[n]))

print("\n=== EDITED TENSORS ===")
for n in edited():
    print(n, tuple(load(n).shape))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 255: restart interface recovery\n"
            "Recovered architecture, record-derived causal notes, and previous-bin "
            "cancellation interfaces before a reversible validation trial. No weights changed.\n")
```