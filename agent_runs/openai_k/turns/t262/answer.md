The diagnostic stopped because I omitted the helper’s required action argument; it produced no replay result and changed no weights. I’ll rerun it with the recorded action vectors. Matching replay and readout outputs will support using this helper for calibration; a mismatch will need resolving first.

```python
r234_capture = {}
sys.setprofile(r234_profile)
try:
    r235_result = r186_heads(r234_indices, rr33_u_all[r234_indices])
finally:
    sys.setprofile(None)

print("=== CAPTURED REPLAY LOCALS ===")
for r235_fn, r235_loc in r234_capture.items():
    print(r235_fn)
    print({k: (type(v).__name__, tuple(v.shape) if hasattr(v, "shape") else
               len(v) if isinstance(v, (list, tuple, dict)) else str(v)[:60])
           for k, v in r235_loc.items() if not isinstance(v, types.FunctionType)})
    for r235_key in ("numeric49", "numeric", "z", "E", "h", "out"):
        if r235_key in r235_loc and hasattr(r235_loc[r235_key], "shape"):
            r235_v = r235_loc[r235_key]
            r235_a = r235_v.detach().numpy() if isinstance(r235_v, torch.Tensor) else np.asarray(r235_v)
            print(r235_key, "range", float(np.min(r235_a)), float(np.max(r235_a)),
                  "sample", r235_a.reshape(-1)[:12].tolist())
print("=== RESULT ===")
if isinstance(r235_result, dict):
    print({k: (type(v).__name__, getattr(v, "shape", None)) for k, v in r235_result.items()})
elif isinstance(r235_result, (list, tuple)):
    print([(type(v).__name__, getattr(v, "shape", None)) for v in r235_result])
else:
    print(type(r235_result).__name__, getattr(r235_result, "shape", None))
print("context151 indices", np.asarray(indices151)[:12])
for r235_name in ("base228", "fact230", "trials230", "target230", "originalweights151", "currentweights151"):
    r235_obj = globals().get(r235_name)
    print(r235_name, type(r235_obj).__name__,
          list(r235_obj)[:15] if isinstance(r235_obj, dict)
          else [(type(v).__name__, getattr(v, "shape", None)) for v in r235_obj]
          if isinstance(r235_obj, (tuple, list)) else getattr(r235_obj, "shape", None))

print("=== PRELUDE PARAMETERS ===")
for r235_name in NAMES:
    if any(s in r235_name for s in ("core.prelude", "core.c_out", "core.delta_in", "core.a_in")):
        print(r235_name, SHAPES[r235_name])
with open("/work/notes.md", "a") as r235_f:
    r235_f.write("\n## Runs 234-235: explicit replay capture\n"
                 "- Run 234 omitted the helper's required action argument and stopped; no weights changed.\n"
                 "- Run 235 supplies recorded actions and captures core/readout internals.\n"
                 "- Python runs: 235; forward passes: 17.\n")
```
