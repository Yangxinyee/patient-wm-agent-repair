The output was truncated by a function’s large default argument, before the useful tensor and cache summaries appeared. Nothing has changed in the model. I’ll repeat the inventory without printing defaults, then use it to identify a testable action-to-reading correction.

```python
print("=== COMPACT FUNCTION INDEX ===")
for rec_name, rec_obj in list(globals().items()):
    if (not rec_name.startswith("rec_") and inspect.isfunction(rec_obj)
            and rec_obj.__module__ == "__main__"):
        rec_c = rec_obj.__code__
        print(rec_name, rec_c.co_varnames[:rec_c.co_argcount])

print("\n=== DICTIONARIES ===")
print([(n, len(v)) for n, v in list(globals().items())
       if isinstance(v, dict) and not n.startswith("rec_")])

print("\n=== NUMPY CACHE ARRAYS ===")
print([(n, v.shape) for n, v in list(globals().items())
       if isinstance(v, np.ndarray) and not n.startswith("rec_")
       and v.size >= 500])

print("\n=== RELEVANT TENSOR SHAPES ===")
for rec_name in NAMES:
    if rec_name.startswith((
        "encode_action.", "core.z_post.", "core.z_prior.", "core.z_in.",
        "core.z_head.", "attr_heads.value_bin.", "last_numeric."
    )):
        print(rec_name, SHAPES[rec_name])

print("\n=== SAVED STATES ===")
for rec_root, rec_dirs, rec_files in os.walk("/work"):
    for rec_file in rec_files:
        if rec_file.endswith(".npz"):
            print(os.path.join(rec_root, rec_file))
```