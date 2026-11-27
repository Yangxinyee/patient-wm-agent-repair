The helpers are still available, but the inspection stopped before revealing their scales and outputs; no weights changed. I’ll use a narrower inspection to recover those details and the value-head interface. This will show whether we can measure a marker’s effect on predicted readings with controlled finite differences rather than relying on hidden-state changes.

```python
import numpy as np
import torch
import inspect

def describe219(obj):
    if isinstance(obj, (np.ndarray, torch.Tensor)):
        return f"{type(obj).__name__}{tuple(obj.shape)}"
    if isinstance(obj, dict):
        return "dict keys=" + str(list(obj.keys())[:12])
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} length={len(obj)}"
    if isinstance(obj, (int, float, str, bool)):
        return repr(obj)[:120]
    return type(obj).__name__

print("=== TRANSITION INPUTS AND OUTPUTS ===")
for nm219 in [
    "rx_tokens126", "rx_hflow", "rx_mem", "rx_ages", "rx_memvalid",
    "rx_W", "rx_layout", "local214", "lc_dims", "lc_zin_base",
    "lc_zbias_base", "relay206", "relay207_base"
]:
    print(nm219, describe219(globals().get(nm219)))
print("transition constants:",
      [x for x in transition214.__code__.co_consts
       if isinstance(x, (str, float, int))])
for kk219, vv219 in local214.items():
    print("local", kk219, describe219(vv219))
    if isinstance(vv219, dict):
        print({k: describe219(v) for k, v in vv219.items()})
    elif isinstance(vv219, (list, tuple)):
        print([describe219(v) for v in vv219])

print("\n=== MARKER SCALES ===")
print("A215", np.asarray(A215))
print("V215 row norms", np.linalg.norm(np.asarray(V215), axis=1))
print("C215 column norms", np.linalg.norm(np.asarray(C215), axis=0))
print("U215 column norms", np.linalg.norm(np.asarray(U215), axis=0))
print("reserved latent coordinates", np.asarray(lc_dims))
for kk219, vv219 in marker215.items():
    print(kk219, describe219(vv219))
ex219 = next(iter(marker215["cf"].values()))
print("counterfactual item:", describe219(ex219))
if isinstance(ex219, dict):
    print({k: describe219(v) for k, v in ex219.items()})
elif isinstance(ex219, (tuple, list)):
    print([describe219(v) for v in ex219])

print("\n=== AVAILABLE LOCAL HEAD FUNCTIONS ===")
for nm219, ob219 in list(globals().items()):
    if inspect.isfunction(ob219) and ob219.__module__ == "__main__" and (
        nm219.startswith(("rx_", "ce_", "r140", "r139"))
        or any(s in nm219.lower() for s in ["head", "value", "predict", "view"])
    ):
        print(nm219, ob219.__code__.co_varnames[:ob219.__code__.co_argcount])

print("\n=== RELEVANT CACHE / ALIGNMENT OBJECTS ===")
for nm219, ob219 in list(globals().items()):
    if nm219.startswith(("rx_", "ce_")) and isinstance(
        ob219, (np.ndarray, torch.Tensor)
    ):
        print(nm219, tuple(ob219.shape))
print("active edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Runs 218–219\n"
            "Run 218 inspection stopped on a numpy-module shape attribute; "
            "no edits changed. Run 219 uses narrow, type-checked inspection "
            "to recover transition outputs, marker scales, and head helpers. "
            "Counts: 219 Python runs, 20 forwards.\n")
```