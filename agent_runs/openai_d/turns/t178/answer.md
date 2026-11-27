The cached edit uses two head neurons for the insulin correction, and the loss scan points to early readings and rare outliers as important problems. I’ll check how the new gate is represented before installing it. The hypothesis is that it suppresses early factual errors while leaving hour-13-to-24 corrections essentially unchanged.

```python
import inspect
import numpy as np

def r156_short(x, depth=0):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return {"shape": x.shape, "min": float(np.min(x)),
                "max": float(np.max(x)), "norm": float(np.linalg.norm(x))}
    if isinstance(x, dict):
        return {str(k): r156_short(v, depth + 1)
                for k, v in list(x.items())[:10]} if depth < 2 else list(x)[:10]
    if isinstance(x, (tuple, list)):
        return [r156_short(v, depth + 1) for v in x[:6]] if depth < 2 else len(x)
    if isinstance(x, (str, int, float, np.number)):
        return x
    return type(x).__name__

print("=== PROMISING GATES ===")
for _key in [(2, 0.1, 3.0), (2, 0.1, 6.0)]:
    print(_key, r156_short(r152_gates[_key]))
print("=== INSTALLED INSULIN OUTLIERS ===")
print(r151_diag.to_string(index=False))

print("=== EXACT LOCAL-EVALUATION HELPERS ===")
for _name in ["r145_hidden", "r145_head", "r138_nll", "r138_pair", "r151_gelu"]:
    _f = globals()[_name]
    try:
        print(_name, inspect.getsource(_f)[:1200])
    except (OSError, TypeError):
        print(_name, "global references:", _f.__code__.co_names,
              "constants:", str(_f.__code__.co_consts)[:500])

print("=== RELEVANT CACHE SHAPES ===")
for _name in sorted(list(globals())):
    if not _name.startswith(("r104", "r106", "r138", "r144", "r145")):
        continue
    _x = globals()[_name]
    if hasattr(_x, "shape"):
        print(_name, type(_x).__name__, tuple(_x.shape))
    elif isinstance(_x, dict):
        print(_name, "dict keys", list(_x)[:12])
    elif isinstance(_x, (int, float, np.number)):
        print(_name, float(_x))

print("=== INSULIN PAIR GEOMETRY ===")
_W = r115_strength2
_W1 = _W["attr_heads.value_bin.0.weight"].detach().cpu().numpy()
_b1 = _W["attr_heads.value_bin.0.bias"].detach().cpu().numpy()
_W2 = _W["attr_heads.value_bin.2.weight"].detach().cpu().numpy()
print("slots", r151_slots)
print("row sum / difference norms",
      np.linalg.norm((_W1[r151_slots[0]] + _W1[r151_slots[1]]) / 2),
      np.linalg.norm((_W1[r151_slots[0]] - _W1[r151_slots[1]]) / 2))
print("bias sum / difference",
      (_b1[r151_slots[0]] + _b1[r151_slots[1]]) / 2,
      (_b1[r151_slots[0]] - _b1[r151_slots[1]]) / 2)
print("output column cancellation",
      float(np.max(np.abs(_W2[:, r151_slots[0]] + _W2[:, r151_slots[1]]))))
```
