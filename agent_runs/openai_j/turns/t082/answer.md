The caches contain controlled dose contrasts at the posterior, but not yet at the value readout. I’ll reconstruct one recurrent step and check it against saved states before using it to design an edit. Close agreement on action bundles would support the method; a substantial mismatch would make its dose contrasts unreliable.

```python
print("RELEVANT CACHES:")
for name, obj in list(globals().items()):
    if not any(s in name.lower() for s in ["atom", "bundle", "feature", "post_", "head_", "state_"]):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, dict):
        print(name, "dict", list(obj.keys())[:4])

print("\nHEAD HELPER RETURN:")
small_head = reconstruct_reading_head(head_meta.iloc[:3])
if isinstance(small_head, dict):
    for key, value in small_head.items():
        print(key, type(value).__name__, getattr(value, "shape", None))
elif isinstance(small_head, (tuple, list)):
    for j, value in enumerate(small_head):
        print(j, type(value).__name__, getattr(value, "shape", None))
else:
    print(type(small_head).__name__, getattr(small_head, "shape", None))

print("\nFUNCTION IMPLEMENTATION METADATA:")
for function_name in ["reconstruct_reading_head", "latent_context_at",
                      "repair_atom_vectors", "repair_set_encode"]:
    function_obj = globals()[function_name]
    print(function_name)
    print("  names:", function_obj.__code__.co_names)
    print("  variables:", function_obj.__code__.co_varnames)
    print("  constants:", [
        c for c in function_obj.__code__.co_consts
        if isinstance(c, (str, int, float, tuple)) and len(str(c)) < 250
    ])

print("\nCOUNTERFACTUAL CACHE VALUE:")
one_cf = post_cf[("insulin", 2.0)]
print(type(one_cf).__name__, getattr(one_cf, "shape", None))
if isinstance(one_cf, dict):
    print([(k, type(v).__name__, getattr(v, "shape", None)) for k, v in one_cf.items()])

print("\nSTATE TIME EXAMPLE:")
print(list(zip(SSID[:12].tolist(), STIME[:12].tolist())))
print("\nATOM-VECTOR HELPER DEFAULTS:")
print(repair_atom_vectors.__defaults__)
print(repair_set_encode.__defaults__)
```