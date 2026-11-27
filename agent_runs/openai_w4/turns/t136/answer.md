The retained kernels suggest delayed insulin effects and shorter norepinephrine effects, but their extrapolated tails remain provisional. I’ll check the decoder’s shape and the perturbation wrapper before connecting it. The connection must reproduce the earlier finite-difference response without amplifying leakage excessively.

```python
from pathlib import Path

print("=== EXACT CONNECTION INTERFACES ===")
for _name122 in ["rp_linear", "linear_repair", "transition73",
                 "transition_with_b110", "probe_means95",
                 "capture_ep112", "shifted_linear118"]:
    _fn122 = globals()[_name122]
    _co122 = _fn122.__code__
    print(_name122)
    print("  globals:", _co122.co_names)
    print("  constants:", repr(_co122.co_consts)[:1000])

print("\n=== DECODER / CAPTURE ===")
dec122 = np.asarray(chosen_channel116["decoder"])
print("decoder", dec122.shape, "norm", np.linalg.norm(dec122))
for _k122, _v122 in chosen_channel116["capture"].items():
    _a122 = np.asarray(_v122)
    print(_k122, _a122.shape, _a122.dtype,
          "range", float(np.min(_a122)), float(np.max(_a122)))
print("factual_rmse", chosen_channel116["factual_rmse"],
      "leakage", chosen_channel116["leakage"])

print("\n=== RELEVANT TENSORS ===")
for _name122 in ["core.b_in.weight", "core.delta_in",
                 "core.prelude_norm.weight", "core.prelude_norm.bias"]:
    _a122 = np.asarray(load(_name122))
    print(_name122, _a122.shape,
          "norm", np.linalg.norm(_a122),
          "first", _a122.reshape(-1)[:5])

print("\n=== RESPONSE DIRECTIONS ===")
for _k122, _v122 in directions118.items():
    print(_k122, np.asarray(_v122).shape,
          "norm", np.linalg.norm(np.asarray(_v122)))

print("\n=== EARLIER IDENTIFICATION AND SUSTAINED-EFFECT NOTES ===")
print(Path("/work/notes.md").read_text()[3200:7400])
```
