The inventory shows that we have reusable action encoders and a local recurrent step, but the output was again truncated. I’ll inspect only those components and the saved-state containers now. My working hypothesis is that a small edit to the action-to-state pathway could restore dose sensitivity without disturbing the last-reading pathway. That needs a verified local sensitivity test before any weight change.

```python
import dis, inspect, re, numpy as np, torch

print("=== TARGETED CONTAINERS ===")
for name, obj in sorted(list(globals().items())):
    if not re.search(r"(state|record|train|tune|bundle|^s[AB0-9]|^st)", name, re.I):
        continue
    if isinstance(obj, dict):
        print(name, "dict", len(obj), "keys", list(obj)[:8])
    elif isinstance(obj, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(name, type(obj).__name__, tuple(obj.shape))

print("\n=== FUNCTION IMPLEMENTATIONS ===")
for fname in ["wm_step_saved", "wm_lin", "wm_mlp", "posterior_features4", "action_vectors4"]:
    f = globals()[fname]
    print("\n", fname, "GLOBAL NAMES:", f.__code__.co_names)
    try:
        text = inspect.getsource(f)
    except Exception:
        text = " ; ".join(
            f"{i.opname} {i.argrepr}" for i in dis.get_instructions(f)
            if i.opname not in {"RESUME", "CACHE", "NOP"}
        )
    print(text[:4500] if fname == "wm_step_saved" else text[:1800])

print("\n=== LATENT/ENCODER TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("z_post.", "z_in.", "encode_action.", "z_head.")):
        print(name, SHAPES[name])

print("\n=== DIAGNOSTIC OBJECTS ===")
for name in ["dyn_estimates", "dynamic_alternatives", "measurement_unit_local",
             "head_layout_spec", "nuisance_models"]:
    obj = globals().get(name)
    print(name, obj if name != "nuisance_models" else
          {k: type(v).__name__ for k, v in obj.items()})
print("Edits:", edited())
```
