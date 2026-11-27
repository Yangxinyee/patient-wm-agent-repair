The earlier latent-routing experiments failed even when their channels were active, so I won’t rely on those linear calibrations. I’ll capture the working replay’s intermediate tensors and value-head inputs directly. This will test whether a factual-quiet feature exists at the recurrent boundary and expose any remaining readout mismatch.

```python
import sys
import pandas as pd

print("=== CONTEXT AND INDEX OBJECTS ===")
for r234_name in ("context151", "r49_readouts", "num_spec0", "r49_centres"):
    r234_obj = globals().get(r234_name)
    if isinstance(r234_obj, dict):
        print(r234_name, {k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else str(v)[:80])
                            for k, v in r234_obj.items()})
    elif isinstance(r234_obj, types.FunctionType):
        print(r234_name, "args", r234_obj.__code__.co_varnames[:r234_obj.__code__.co_argcount],
              "globals", r234_obj.__code__.co_names)
    else:
        print(r234_name, type(r234_obj).__name__, getattr(r234_obj, "shape", None))
print("151 objects", [n for n in globals() if "151" in n])
print("228-231 objects", [n for n in globals() if any(s in n for s in ("228", "229", "230", "231"))
                         and not n.startswith(("r232", "r233", "r234"))][:100])

r234_indices = np.flatnonzero(
    (np.asarray(rr33_hours) > 0) &
    (np.asarray(rr33_stays) == np.roll(np.asarray(rr33_stays), 1))
)[:8]
r234_capture = {}
r234_codes = {globals()[n].__code__: n for n in
              ("r59_core", "repaired_core_local", "r49_readouts", "r179_capture_head")
              if isinstance(globals().get(n), types.FunctionType)}
def r234_profile(frame, event, arg):
    if event == "return" and frame.f_code in r234_codes:
        r234_capture[r234_codes[frame.f_code]] = dict(frame.f_locals)
sys.setprofile(r234_profile)
try:
    r234_result = r186_heads(r234_indices)
finally:
    sys.setprofile(None)

print("\n=== CAPTURED REPLAY LOCALS ===")
for r234_fn, r234_loc in r234_capture.items():
    print(r234_fn)
    print({k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else
               len(v) if isinstance(v, (list, tuple, dict)) else str(v)[:70])
           for k, v in r234_loc.items() if not isinstance(v, types.FunctionType)})
print("head result",
      {k: (getattr(v, "shape", None), np.asarray(v).ravel()[:8].tolist())
       for k, v in r234_result.items()} if isinstance(r234_result, dict)
      else (type(r234_result).__name__, getattr(r234_result, "shape", None)))

print("\n=== SCHEMA AND PRELUDE PARAMETERS ===")
print(Path("/data/README.md").read_text()[:5000])
for r234_name in NAMES:
    if any(s in r234_name for s in ("core.prelude", "core.c_out", "core.delta_in", "core.a_in")):
        print(r234_name, SHAPES[r234_name])
with open("/work/notes.md", "a") as r234_f:
    r234_f.write("\n## Run 234: capture working replay internals\n"
                 "- Earlier latent calibration was refuted by official rollouts.\n"
                 "- Capturing generic core and head inputs to avoid further surrogate-only edits.\n"
                 "- No weights changed. Python runs: 234; forward passes: 17.\n")
```
