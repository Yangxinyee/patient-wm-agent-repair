The update formula follows the architecture, and including the current bundle in memory gave the closest match. One possible missing input is infusion END atoms: these can share a bundle with a new infusion. I’ll check atom counts alongside the errors and inspect the lab encoding. Errors concentrated in END-containing bundles would support that explanation.

```python
import pandas as pd
import numpy as np

print("=== EMBEDDING TABLES AND CONFIG ===")
print(VOCAB["config"])
for name in rm_W:
    if name.startswith("embed.categorical") or name.startswith("encode."):
        print(name, tuple(rm_W[name].shape))

print("\n=== RECONSTRUCTION ARRAY SHAPES ===")
for name in sorted(globals()):
    if name.startswith(("rm_", "rec_w", "rec_atom", "rec_numbers", "rec_e", "rec_counts")):
        obj = globals()[name]
        if hasattr(obj, "shape") and not callable(obj.shape):
            print(name, tuple(obj.shape))
        elif isinstance(obj, list):
            print(name, "list", len(obj))
        elif isinstance(obj, dict) and name != "rm_W":
            print(name, "dict", list(obj)[:10])

print("\n=== WORLD ROWS ===")
print(type(rec_world_rows))
if isinstance(rec_world_rows, pd.DataFrame):
    print(rec_world_rows.head(3).to_string(index=False))
    print(rec_world_rows[rec_world_rows.event == "glucose_lab"].head(3).to_string(index=False))
else:
    print(repr(rec_world_rows)[:1000])

print("\n=== TRANSITION ERROR VERSUS BUNDLE COMPOSITION ===")
td = rec_transition_diag.copy()
lookup = {
    (int(s), round(float(h), 4)): i
    for i, (s, h) in enumerate(zip(rm_states["stay_id"], rm_states["hour"]))
}
idx = np.array([lookup[(int(r.stay_id), round(float(r.hour), 4))]
                for r in td.itertuples()], dtype=int)
for field in ["n_atoms", "has_insulin", "has_norepinephrine"]:
    td[field] = np.asarray(rm_states[field])[idx]
print(td.to_string(index=False))

print("\n=== FIRST STAY BUNDLE COUNTS ===")
ii = np.flatnonzero(rm_states["stay_id"] == rm_states["stay_id"][0])[:25]
print(pd.DataFrame({k: np.asarray(rm_states[k])[ii]
                    for k in ["stay_id", "hour", "n_atoms", "has_insulin",
                              "has_norepinephrine"]}).to_string(index=False))
```
