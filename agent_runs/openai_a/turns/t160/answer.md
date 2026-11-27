The validated edit is backed up. The remaining reconstruction error is small relative to the slot magnitudes, but that alone does not establish that dose contrasts are accurate. I’ll identify the cached action inputs and state indices, then compare changed-dose transitions against the unchanged-dose baseline.

```python
import dis

print("=== HELPER DEPENDENCIES ===")
for function_name in [
    "rm_insulin_atoms", "r131_step", "rec_context",
    "cpu_head_parts", "reading_metadata", "rec_memory"
]:
    f = globals()[function_name]
    print("\n", function_name)
    print("globals:", f.__code__.co_names)
    print("constants:", repr(f.__code__.co_consts)[:1300])
    print("locals:", f.__code__.co_varnames)

print("\n=== RECONSTRUCTION INPUTS ===")
for name, obj in list(globals().items()):
    if not name.startswith(("rm_", "r131_", "r132_", "r138_", "r139_")):
        continue
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, tuple(obj.shape))
    elif isinstance(obj, pd.DataFrame):
        print(name, obj.shape, list(obj.columns)[:8])
    elif isinstance(obj, dict):
        if name != "rm_W":
            print(name, "keys", list(obj.keys())[:8])
    elif isinstance(obj, (str, int, float, bool)):
        print(name, repr(obj)[:160])

print("\n=== STATE ARRAYS AND FILES ===")
for name, obj in list(globals().items()):
    if isinstance(obj, (np.ndarray, torch.Tensor)) and len(obj.shape) == 2:
        if obj.shape[1] == 992:
            print(name, tuple(obj.shape))
    elif isinstance(obj, np.lib.npyio.NpzFile):
        print(name, obj.files)
print("state files:", [str(p) for p in Path("/work").glob("fwd_*/*states*.npz")])

print("\n=== LATENT AND RECURRENT PREFIXES ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_prior", "delta_in", "a_in", "b_in", "c_out")):
        print(name, SHAPES[name])
print("\nVOCAB keys:", list(VOCAB.keys()))
```
