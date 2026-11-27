The candidate coefficients and installation helpers are available. The safer `fw=10, gain=6` candidate has local factual-loss increases of +0.00187 on held-out training stays and +0.00317 on tuning stays, leaving more margin than `gain=8`. I’ll verify how the helper preserves the insulin correction and recover the causal targets. This candidate is worth testing only if its stronger MAP response moves toward those targets without breaking factual fit.

```python
import dis, inspect, numpy as np, pandas as pd
from pathlib import Path

print("CANDIDATE")
c196 = map192_candidates[(10.0, 6.0)]
print(type(c196).__name__, getattr(c196, "shape", None))
if isinstance(c196, np.ndarray):
    print("norm", np.linalg.norm(c196), "finite", np.isfinite(c196).all())

print("\nINSTALLATION HELPER")
print("global names:", head177_make.__code__.co_names)
try:
    print(inspect.getsource(head177_make))
except Exception:
    print("\n".join(f"{i.opname} {i.argrepr}"
                    for i in dis.get_instructions(head177_make))[:4300])

print("\nEVALUATOR GLOBALS", head177_eval.__code__.co_names)
for name in set(head177_make.__code__.co_names):
    if name in globals():
        obj = globals()[name]
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict", list(obj)[:10])
        elif isinstance(obj, (str, int, float, list, tuple)):
            print(name, str(obj)[:180])

print("\nNOTEBOOK HEADINGS")
nt196 = Path("/work/notes.md").read_text()
headings196 = [line for line in nt196.splitlines() if line.startswith("## ")]
print("\n".join(headings196[:38]))

print("\nKERNEL TARGETS")
print(pd.read_csv("/work/kernel_targets.csv").to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart audit, runs 193–196\n"
            "- Re-read architecture; all active edits remain in the four value-head tensors.\n"
            "- Recovered calibrated MAP frontier and installation helper. No new edits installed.\n"
            "- Considering fw=10, gain=6: local held-out training dNLL +0.00187, tuning +0.00317.\n"
            "- 19 forward passes used; combined edit from forward 19 remains the fallback.\n")
```
