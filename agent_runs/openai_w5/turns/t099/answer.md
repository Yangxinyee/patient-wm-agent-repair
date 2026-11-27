The original rollout shifts are small, and the recurrent block has a direct path from the action token to state. I’ll measure that path with fixed states and shared readout randomness. If it has a stable response direction, we can test a targeted edit; if not, simply amplifying the existing action signal is unlikely to work.

```python
import pandas as pd, numpy as np

for nm in ["local_step67", "prelude_all83", "next_means_crn74",
           "action_atom_exact", "wm_attention", "wm_layer"]:
    f = globals()[nm]
    print(nm, "GLOBALS:", f.__code__.co_names)
    print("CONSTANTS:", [x for x in f.__code__.co_consts
                         if isinstance(x, (str, int, float, tuple))][:35])

print("\nCONFIG", VOCAB["config"])
print("\nSTATE/CACHE OBJECTS")
for nm in ["s4", "cacheTrain", "wm_e4", "U4", "W", "W0", "weights",
           "E83", "Ebase83", "Ecf83", "cf83", "fit83", "fits83",
           "probe83", "rows83", "DT4", "age67"]:
    if nm in globals():
        ob = globals()[nm]
        if isinstance(ob, dict):
            print(nm, "dict", list(ob.keys())[:12])
        else:
            print(nm, type(ob).__name__, getattr(ob, "shape", None))

print("\nORIGINAL SHIFTS")
for pth in sorted(Path("/work").glob("fwd_*/dose_edit_original.csv")):
    dd = pd.read_csv(pth)
    print(str(pth), dd.loc[dd.reached].groupby("edit")["shift"].mean().round(4).to_dict())

print("\nREGISTER/HEAD SHAPES")
for n in NAMES:
    if any(s in n for s in ["attr_heads.value_bin", "last_numeric", "embed_transition.value",
                            "core.out_norm", "query_norm", "z_head"]):
        print(n, SHAPES[n])
print("Current edits", edited())
```
