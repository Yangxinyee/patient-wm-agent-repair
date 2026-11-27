The reusable objects are present, but the inventory was too long to expose the relevant routines. I’ll narrow the inspection to the recurrent step, action encoder, corrected readout, and original rollout results. This should show whether a small encoder edit can carry a stable drug signal into future readings.

```python
import inspect, dis, io, contextlib, pandas as pd

def show_impl85(name, limit=2200):
    f = globals()[name]
    print("\nFUNCTION", name)
    try:
        print(inspect.getsource(f)[:limit])
    except Exception:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dis.dis(f, depth=0)
        print(buf.getvalue()[:limit])

for nm in ["wm_layer", "wm_set", "local_step67", "next_means_crn74", "prelude_all83"]:
    show_impl85(nm, 1500)

print("\n=== RELEVANT TENSOR SHAPES ===")
for n in NAMES:
    if any(s in n for s in ["encode_action", "core.prelude", "core.b_in", "core.c_out",
                            "core.delta_in", "core.a_in", "core.flow", "core.coda",
                            "attr_heads.value_bin", "last_numeric"]):
        print(n, SHAPES[n])

print("\n=== ORIGINAL ROLLOUT SUMMARY ===")
for pth in sorted(Path("/work").glob("fwd_*/dose_edit_original.csv")):
    dd = pd.read_csv(pth)
    print(str(pth), dd.shape, dd.head(5).to_dict("records"))

print("\n=== RECENT OBJECT SHAPES ===")
for nm in ["E83", "Ebase83", "Ecf83", "fits83", "action_u83", "rows83",
           "D83", "R83", "prelude83", "weights", "W0"]:
    if nm in globals():
        ob = globals()[nm]
        print(nm, type(ob).__name__, getattr(ob, "shape", None),
              list(ob.keys())[:8] if isinstance(ob, dict) else "")
print("Edits:", edited())
```
