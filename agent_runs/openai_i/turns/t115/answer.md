The architecture confirms that drug actions enter the recurrent state separately from measurements, so the repair need not directly alter the measurement heads. I’ll inspect the retained features and regression helpers to find a dose-sensitive correction that is small on factual records. A useful feature must respond to changed doses, not merely correlate with patient condition.

```python
import inspect, numpy as np, pandas as pd

print("=== ARRAY AND TABLE INVENTORY ===")
items_snapshot = list(globals().items())
inventory = []
for name, obj in items_snapshot:
    if name.startswith("_") or name in {"obj", "value"}:
        continue
    if isinstance(obj, (np.ndarray, pd.DataFrame, pd.Series)):
        inventory.append(f"{name}: {type(obj).__name__}{tuple(obj.shape)}")
print("\n".join(inventory)[:4700])

print("\n=== HELPERS ===")
helpers = []
for name, obj in items_snapshot:
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        try:
            helpers.append(f"{name}{inspect.signature(obj)}")
        except Exception:
            helpers.append(name)
print("\n".join(helpers)[:2600])

print("\n=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in ["z_post", "z_in", "encode_action", "value_bin", "numeric"]):
        print(name, SHAPES[name])

print("\n=== RECENT LATENT-REGRESSION OBJECTS ===")
for name, obj in items_snapshot:
    if any(s in name.lower() for s in ["direction", "latent", "reg104", "coef104"]):
        if isinstance(obj, dict):
            print(name, list(obj.keys())[:12])
        elif hasattr(obj, "shape"):
            print(name, tuple(obj.shape))
        elif isinstance(obj, (str, int, float, tuple, list)):
            print(name, str(obj)[:180])

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 106 — retained-feature inventory\n"
            "- Separate action and observation pathways permit a recurrent rather than sensor-specific repair.\n"
            "- Inspecting retained arrays and helper contracts to select a dose-sensitive, factual-centered feature.\n"
            "- No weights changed.\n")
```