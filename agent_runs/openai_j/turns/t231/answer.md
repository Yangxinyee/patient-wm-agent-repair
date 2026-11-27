The architecture shows that drug doses affect the shared recurrent state, while the active edits change only the value head. The notebook also shows why simple dose–reading associations are unreliable here: MAP estimates change substantially when history is controlled. I’ll recover the selective-edit helpers and causal target summary. This will let me test whether the current head correction is too weak without disturbing other prediction heads.

```python
import inspect, dis
import numpy as np

print("=== CURRENT EDIT HELPERS ===")
for n205 in ("head177_make", "head177_eval"):
    f205 = globals().get(n205)
    if f205 is None:
        continue
    print(n205, inspect.signature(f205))
    print("Referenced names:", f205.__code__.co_names)
    try:
        print(inspect.getsource(f205))
    except (OSError, TypeError):
        if n205 == "head177_make":
            dis.dis(f205)

print("\n=== OTHER RELEVANT HELPER SIGNATURES ===")
for n205, o205 in list(globals().items()):
    if inspect.isfunction(o205) and any(
        s in n205 for s in ("190", "192", "172", "contrast", "kernel")
    ):
        print(n205, inspect.signature(o205), o205.__code__.co_names)

print("\n=== MAP CANDIDATE REPRESENTATION ===")
c205 = map192_candidates[(10, 6)]
print(type(c205), getattr(c205, "shape", None))
if isinstance(c205, dict):
    for k205, v205 in c205.items():
        print(k205, type(v205).__name__, getattr(v205, "shape", ""),
              repr(v205)[:200] if not hasattr(v205, "shape") else "")
else:
    print(repr(c205)[:600])

print("\n=== HEAD177 VARIABLES ===")
for n205, o205 in list(globals().items()):
    if n205.startswith("head177") and not inspect.isfunction(o205):
        print(n205, type(o205).__name__, getattr(o205, "shape", ""),
              list(o205)[:20] if isinstance(o205, dict) else
              (repr(o205)[:160] if not hasattr(o205, "shape") else ""))

print("\n=== RECORD-DERIVED TARGETS ===")
print(Path("/work/kernel_targets.csv").read_text()[:1800])

print("\n=== EARLY CAUSAL EVIDENCE ===")
p205 = notes204.find("## Factual baseline and latent dynamics")
print(notes204[p205:p205+3500])
```
