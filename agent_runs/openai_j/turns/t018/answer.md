Dose and rate have dedicated numeric embedding columns, while the value head is a small two-layer network. That offers possible edit points, but I still need to check whether the saved states let me reconstruct its inputs. If they do, I can compute a constrained regression edit rather than alter the shared dynamics blindly.

```python
print("=== OTHER HEAD / CONTEXT TENSORS ===")
for name in NAMES:
    if (name.startswith(("gap.", "out_norm.", "pending_", "filter_", "time_head.", "z_prior."))
        or name in {"embed.kq.weight", "last_numeric.weight", "last_status.weight"}):
        print(name, SHAPES[name])

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
    for path in folder.glob("*.npz"):
        with np.load(path) as archive:
            print(path.name, {k: archive[k].shape for k in archive.files})
            for k in archive.files:
                if archive[k].ndim == 1:
                    print(k, archive[k][:4])
    for path in folder.glob("*.csv"):
        frame = pd.read_csv(path)
        print(path.name, frame.shape, list(frame.columns))
        print(frame.head(1).to_string(index=False))

print("\n=== PANEL DETAILS ===")
print("HOURS:", HOURS)
print("train IDs:", train_ids[:8], "tuning IDs:", tuning_ids[:8])
print("panel index:", panel.index.names)
print(panel.head(4).to_string())
print("training sample:")
print(train.head(12).to_string(index=False))

print("\n=== RECENT ANALYSIS OBJECTS ===")
for key in ["kernel_fits", "kernels", "kernel_results", "kernel_table",
            "kernel_targets", "targets", "iv_table", "sensor_iv_table",
            "HX", "innov", "HF"]:
    if key in globals():
        obj = globals()[key]
        if isinstance(obj, dict):
            print(key, repr(obj)[:900])
        elif isinstance(obj, pd.DataFrame):
            print(key, obj.head(4).to_string(index=False))
        elif hasattr(obj, "shape"):
            print(key, obj.shape)

print("\n=== LAST-NUMERIC COLUMN NORMS ===")
print(np.linalg.norm(load_original("last_numeric.weight"), axis=0))
print("\n=== EDIT STATUS ===", edited())
```
