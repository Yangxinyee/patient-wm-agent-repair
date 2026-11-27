The corrected summary shows good separation by reading type: the insulin correction is off for MAP, and the norepinephrine correction is off for glucose and cuff MAP. Both activate on only part of their intended readings. I’ll recover the controlled-dose probe machinery to test whether the decoder responds differently to increases and decreases; that would support revising the decoder rather than merely amplifying it.

```python
from pathlib import Path
import numpy as np, inspect

text189 = Path("/work/notes.md").read_text()
for heading189, limit189 in (
    ("## Run 99:", 1700), ("## Run 124:", 2200)):
    p189 = text189.find(heading189)
    print("\n", heading189, "\n", text189[p189:p189+limit189] if p189>=0 else "not found")

print("\nRECONSTRUCTION / PROBE OBJECTS")
for k189, v189 in list(globals().items()):
    if k189.startswith(("r97_", "r98_", "r99_", "r124_", "r125_", "r129_")):
        if hasattr(v189, "shape"):
            print(k189, type(v189).__name__, tuple(v189.shape))
        elif isinstance(v189, dict):
            print(k189, "dict", list(v189)[:12])
        elif isinstance(v189, (int,float,str)):
            print(k189, repr(v189)[:100])
        elif isinstance(v189, (list,tuple)):
            print(k189, type(v189).__name__, len(v189))

print("\nPROBE FUNCTION DETAILS")
for name189 in ("r124_context", "r129_queries"):
    f189 = globals()[name189]
    try:
        print(name189, inspect.getsource(f189)[:1400])
    except Exception:
        print(name189, "names", f189.__code__.co_names,
              "locals", f189.__code__.co_varnames,
              "constants", [x for x in f189.__code__.co_consts
                            if isinstance(x,(str,int,float,tuple))][:25])

print("\nTRAIN ID RANGE", np.asarray(train_ids)[:5], np.asarray(train_ids)[100:105])
with np.load("/work/fwd_3/states_original.npz", allow_pickle=False) as z189:
    print("STATE FILE KEYS", {k: z189[k].shape for k in z189.files})
print("\nRELATED PROBE VARIABLE NAMES\n", [
    k for k in list(globals())
    if k.startswith(("r118_", "r119_", "r120_", "r121_", "r122_", "r123_", "r126_", "r127_", "r128_"))
][:130])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 189: recover controlled-dose probes\nCorrect gates isolate drug/reading types, but activate on only part of the relevant readings. Investigating dose-embedding asymmetry and recovering existing controlled state-probe machinery before another edit. No weights changed.\n")
```