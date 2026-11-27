The architecture shows that doses can affect the state through several paths, while factual predictions also rely on the last-reading register. That suggests a selective dose-signal edit may work better than removing a whole pathway. I’ll inspect the saved directions and tensor shapes; this approach is promising only if the dose-related directions can be separated from ordinary reading history.

```python
import inspect
import json

print("Candidate B factual means:", factual_b_summary)
print("Current operations:", ops())

print("\n=== KERNEL INVENTORY ===")
inventory = []
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"In", "Out", "obj", "inventory"}:
        continue
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        inventory.append((name, type(obj).__name__, tuple(obj.shape)))
    elif isinstance(obj, dict):
        inventory.append((name, "dict", list(obj)[:8]))
    elif callable(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            inventory.append((name, "function", str(inspect.signature(obj))))
        except Exception:
            inventory.append((name, "function"))
print(repr(inventory)[:5500])

print("\n=== SAVED DIRECTIONS ===")
if "dose_directions" in globals():
    for key, val in dose_directions.items():
        if isinstance(val, dict):
            print(key, {
                k: ({"shape": tuple(v.shape), "norm": float(np.linalg.norm(v))}
                    if isinstance(v, np.ndarray) else str(v)[:180])
                for k, v in val.items()
            })
        else:
            print(key, type(val).__name__, getattr(val, "shape", None))

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if (name.startswith(("embed.numeric", "encode_action", "core.c_out",
                         "core.out_norm", "core.prelude", "attr_heads.value_bin",
                         "last_numeric", "query_norm", "z_head"))
        or name in {"embed.code.weight", "core.b_in.weight"}):
        print(name, SHAPES[name])

print("\n=== VOCABULARY STRUCTURE ===")
print("Keys:", list(VOCAB))
print("Config:", VOCAB.get("config"))
print("Codes:", VOCAB.get("codes"))
for key in VOCAB:
    if "norm" in key.lower():
        print(key, json.dumps(VOCAB[key])[:1500])

print("\n=== STATE FILE STRUCTURE ===")
for p in sorted(Path("/work/fwd_3").glob("*.npz")):
    with np.load(p) as z:
        print(p.name, {k: z[k].shape for k in z.files})
```