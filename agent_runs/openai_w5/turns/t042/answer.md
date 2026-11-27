The architecture confirms that doses can affect both the action token and the latent posterior. The inventory stopped on a harmless introspection error; no weights changed. I’ll inspect the recurrence tensors and helpers next, so the next edit can target a verified pathway rather than rely on another ablation.

```python
import numpy as np
import torch
import inspect

print("=== CORE / ACTION TENSORS ===")
for name in NAMES:
    if name.startswith(("core.", "encode_action.", "z_post.", "z_in.", "z_head.")):
        print(name, SHAPES[name])

print("\n=== OBJECT INVENTORY ===")
for name, obj in list(globals().items()):
    if name.startswith("_") or inspect.ismodule(obj) or inspect.isclass(obj) or callable(obj):
        continue
    if isinstance(obj, dict):
        print(name, "dict", len(obj), list(obj.keys())[:10])
    elif isinstance(obj, (np.ndarray, torch.Tensor)):
        print(name, type(obj).__name__, tuple(obj.shape))
    elif isinstance(obj, pd.DataFrame):
        print(name, "DataFrame", obj.shape, list(obj.columns)[:20])
    elif isinstance(obj, (list, tuple)) and len(obj) < 100:
        print(name, type(obj).__name__, len(obj), repr(obj)[:250])

print("\n=== HELPER SOURCES, WHEN AVAILABLE ===")
for name in ["action_atom_exact", "measurement_atom_local", "set_encode_single_local",
             "context_from_saved", "head_features_layout", "mechanistic_shifts",
             "clustered_ols", "causal_features"]:
    obj = globals().get(name)
    if callable(obj):
        try:
            print("\n" + name + "\n" + inspect.getsource(obj))
        except (OSError, TypeError):
            print(name, "source unavailable", obj.__code__.co_varnames[:obj.__code__.co_argcount])

print("\n=== VOCAB ===")
print(VOCAB)
print("\n=== CURRENT EDITS ===", edited())
```
