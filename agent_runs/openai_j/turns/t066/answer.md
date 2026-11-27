The posterior has a 192-unit hidden layer before its latent output, which offers a more targeted edit than scaling all dose embeddings. I’ll use the saved factual states to test that layer directly, rather than continue reconstructing every observation. The test will check whether it can distinguish changed doses from ordinary factual variation.

```python
import torch
import inspect

print("=== HELPER DEPENDENCIES ===")
needed_objects = set()
for name in ["repair_atom_vectors", "repair_set_encode", "latent_context_at",
             "encode_single_debug", "wt", "mlp", "ln"]:
    obj = globals()[name]
    code = obj.__code__
    print(name, "locals:", code.co_varnames,
          "dependencies:", code.co_names,
          "constants:", [x for x in code.co_consts
                         if isinstance(x, (str, int, float, bool))][:25])
    needed_objects.update(x for x in code.co_names if x in globals())

print("\n=== REFERENCED DATA STRUCTURES ===")
for name in sorted(needed_objects):
    obj = globals()[name]
    if isinstance(obj, pd.DataFrame):
        print(name, obj.shape, obj.columns.tolist())
        print(obj.head(2).to_dict("records"))
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", len(obj), "keys:", list(obj)[:10])
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj))

print("\n=== ADDITIONAL STATE/ATOM OBJECT NAMES ===")
print([name for name in sorted(list(globals()))
       if any(s in name.lower() for s in
              ["atom", "bundle", "posterior", "context", "state", "repair_"])])

print("\n=== SENSOR IV CHECKS ===")
print(pd.read_csv("/work/sensor_iv.csv").to_string(index=False))
print("Active edits:", edited())
```
