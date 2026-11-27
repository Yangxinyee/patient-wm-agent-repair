The controlled estimates are stable across dose ranges: about −2 glucose units per insulin rate unit and +3.3 MAP units per norepinephrine rate unit at the next reading, with effects persisting afterward. I’ll now locate the cached reconstruction functions and action tensors. They will let me test whether an earlier edit can restore that response rather than amplify confounding.

```python
print("=== FUNCTION INDEX ===")
rec_funcs = []
for rec_name, rec_obj in list(globals().items()):
    if (not rec_name.startswith("rec_") and inspect.isfunction(rec_obj)
            and rec_obj.__module__ == "__main__"):
        try:
            rec_funcs.append(rec_name + str(inspect.signature(rec_obj)))
        except Exception:
            rec_funcs.append(rec_name)
print("\n".join(rec_funcs))

print("\n=== DICTIONARY INDEX ===")
print([(n, len(v)) for n, v in list(globals().items())
       if isinstance(v, dict) and not n.startswith("rec_")])

print("\n=== ARRAY INDEX (ANALYSIS CACHES) ===")
rec_arrays = [(n, v.shape) for n, v in list(globals().items())
              if isinstance(v, np.ndarray) and not n.startswith("rec_")
              and v.size >= 500]
print(rec_arrays)

print("\n=== RELEVANT TENSOR SHAPES ===")
for rec_name in NAMES:
    if rec_name.startswith((
        "encode_action.", "core.z_post.", "core.z_prior.", "core.z_in.",
        "core.z_head.", "attr_heads.value_bin.", "last_numeric."
    )):
        print(rec_name, SHAPES[rec_name])

print("\n=== STATE FILES ===")
for rec_root, rec_dirs, rec_files in os.walk("/work"):
    for rec_file in rec_files:
        if rec_file.endswith(".npz"):
            print(os.path.join(rec_root, rec_file))
```