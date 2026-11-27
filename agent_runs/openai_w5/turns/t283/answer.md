The installed head subtracts a linear function of ordinary model features from its dose signal. Those features are not guaranteed to be pre-treatment. I will check whether the last-reading register can provide a replacement center and inspect the hourly rollout errors. An error already present at hour 13 would favor an immediate encoding or centering problem over feedback alone.

```python
import pandas as pd

print("=== EXACT CENTER DIFFERENCES ===")
r248_private = [21, 61, 15, 66]
for r248_control, r248_partner in _pair_for_control.items():
    r248_d = r247_w0[r248_partner] - r247_w0[r248_control]
    r248_o = r248_d.copy()
    r248_o[r248_private] = 0
    print(r248_control, r248_partner,
          "ordinary_norm", float(np.linalg.norm(r248_o)),
          "bias_difference", float(r247_b0[r248_partner] - r247_b0[r248_control]),
          "private", r248_d[r248_private].tolist())

print("\n=== REGISTER AND EMBEDDING CHANNEL SAFETY ===")
for r248_name in ("last_numeric.weight", "last_numeric.bias",
                  "encode.mlp1.0.weight", "encode_action.mlp1.0.weight",
                  "embed.bins.value.weight"):
    r248_t = np.asarray(load(r248_name))
    print(r248_name, r248_t.shape)
    if r248_t.ndim == 2 and r248_t.shape[1] == 192:
        print("  input-pair difference norms:",
              [float(np.linalg.norm(r248_t[:, a] - r248_t[:, b]))
               for a, b in private_pairs])
    elif r248_t.ndim == 2 and r248_t.shape[0] == 192:
        print("  output-pair difference norms:",
              [float(np.linalg.norm(r248_t[a] - r248_t[b]))
               for a, b in private_pairs])

print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(VOCAB))
for r248_key in VOCAB:
    if any(s in r248_key.lower() for s in ("norm", "numeric", "code")):
        print(r248_key, repr(VOCAB[r248_key])[:1800])

print("\n=== AVAILABLE RECORD OBJECTS ===")
for r248_name in ("train", "tuning", "panel", "panel_tune", "A", "B", "ids237"):
    if r248_name in globals():
        r248_obj = globals()[r248_name]
        print(r248_name, type(r248_obj).__name__,
              getattr(r248_obj, "shape", None),
              repr(r248_obj)[:180] if not isinstance(r248_obj, pd.DataFrame) else
              list(r248_obj.columns))

print("\n=== FORWARD FILE INVENTORY ===")
for r248_dir in sorted(Path("/work").glob("fwd_*"),
                       key=lambda p: int(p.name.split("_")[-1])):
    r248_files = [p.name for p in r248_dir.glob("*.csv")]
    if r248_files:
        print(r248_dir.name, r248_files)

for r248_num in (17, 21, 22, 23):
    r248_path = Path(f"/work/fwd_{r248_num}/dose_edit_edited.csv")
    if r248_path.exists():
        r248_df = pd.read_csv(r248_path)
        print("\n", r248_path, "shape", r248_df.shape,
              "columns", list(r248_df.columns))
        print(r248_df.head(3).to_string(index=False))
```
