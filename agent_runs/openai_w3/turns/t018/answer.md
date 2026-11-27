The architecture confirms that doses affect the recurrent state, while value predictions also rely heavily on the last-reading register. I’ll inspect the existing reconstruction functions and relevant weights next; this should reveal whether a targeted state edit can change dose responses without disturbing factual predictions.

```python
import types
items = list(globals().items())
print("=== VARIABLE NAMES ===")
print(" ".join(k for k, v in items if not k.startswith("_")))
print("=== RECOVERY: LATER VARIABLES ===")
start = next((i for i, (k, v) in enumerate(items) if k == "bread"), 0)
for k, v in items[start:]:
    if k.startswith("_"):
        continue
    if isinstance(v, types.FunctionType):
        print("FUNCTION", k, "args/local names:", v.__code__.co_varnames[:18])
    elif hasattr(v, "shape") and not isinstance(v, types.ModuleType):
        print(k, type(v).__name__, v.shape)
    elif isinstance(v, dict):
        print(k, "dict", len(v), "keys:", list(v)[:10])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, len(v))
    elif isinstance(v, (str, int, float, bool)):
        print(k, repr(v)[:100])
print("=== RELEVANT TENSORS ===")
for n in NAMES:
    if n.startswith(("z_", "core.z", "attr_heads.value", "last_numeric", "embed.numeric",
                     "encode_action", "core.out_norm", "core.gap", "core.flow")):
        print(n, SHAPES[n])
print("=== CONFIG AND NUMERIC NORMALIZATION ===")
print(VOCAB["config"])
print(VOCAB["codes"])
print(VOCAB["numeric"])
```