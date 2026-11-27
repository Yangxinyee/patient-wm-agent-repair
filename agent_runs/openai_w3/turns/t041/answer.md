The needed helpers are still available, although their source text is not. The empty-observation mismatch makes set-size handling or bundle selection worth checking before more metadata searches. I’ll inspect those implementations and verify which records each saved state represents; a cardinality mismatch would explain the failed single-reading tests.

```python
import dis, numpy as np, torch

print("=== ENCODER BYTECODE ===")
for name in ("encode_one_set", "clean_single_encode", "isolated_record"):
    fn = globals()[name]
    print("\n", name)
    print(dis.Bytecode(fn).dis()[:2300])

print("\n=== OTHER HELPER CONSTANTS ===")
for name in ("make_atom", "encode_bundles", "state_context", "posterior_mean"):
    fn = globals()[name]
    print(name, "names:", fn.__code__.co_names)
    print("constants:", repr(fn.__code__.co_consts)[:900])

print("\n=== STATE AND RECORD EXAMPLES ===")
print("S keys:", {k: np.asarray(v).shape for k, v in S.items()})
print(state_records.head(12).to_string(index=False))
print("state examples:")
for j in range(min(8, len(S["hour"]))):
    print(j, S["stay_id"][j], S["hour"][j],
          S["has_insulin"][j], S["has_norepinephrine"][j])

print("\n=== REMAINING LIVE ARRAY NAMES ===")
seen_sx = False
for name, obj in list(globals().items()):
    if name == "sx":
        seen_sx = True
    if seen_sx and not name.startswith("_"):
        if hasattr(obj, "shape") and not callable(getattr(obj, "shape", None)):
            print(name, tuple(obj.shape))
        elif isinstance(obj, dict) and len(obj) < 100:
            print(name, "keys", list(obj)[:8])

print("\n=== CACHE CONSISTENCY ===")
for name in ("encode.mlp1.0.weight", "encode.mlp2.0.weight",
             "encode.norm.weight", "embed.numeric.weight"):
    print(name, float((W[name] - load_original(name)).abs().max()))

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 36; forwards 3. Checking set cardinality, bundle alignment, helper bytecode and original-weight cache consistency. Empty-observation mismatch makes preprocessing/cardinality a priority. No edits.\n")
```