Reversing and enlarging the gains still did not recover the supported effects. This suggests the posterior injection disrupts the existing mapping rather than providing useful control. I’ll revert it and inspect the value-head inputs. A more direct edit is worth pursuing only if dose changes leave a measurable signal there that can be amplified without damaging factual predictions.

```python
import numpy as np, inspect
from pathlib import Path

candidate_negative173 = {name: np.asarray(load(name)).copy() for name in edited()}
for name in list(edited()):
    drop_edit(name)
print("Active edits:", edited())

print("\n=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if name.startswith(("attr_heads.value_bin", "last_numeric", "z_prior",
                        "z_head", "type_emb", "query_norm", "core.gap",
                        "gap", "core.z_in")):
        print(name, SHAPES[name])

print("\n=== READOUT AND REPLAY HELPERS ===")
for name in ["readout153", "readout_capture156", "capture159",
             "replay_inputs161", "core_step136", "mlp136"]:
    fn173 = globals()[name]
    print("\nFUNCTION", name)
    try:
        print(inspect.getsource(fn173))
    except Exception:
        print("Referenced globals:", fn173.__code__.co_names)
        print("Constants:", repr(fn173.__code__.co_consts)[:1900])

print("\n=== DATAFRAMES ===")
for name, obj in list(globals().items()):
    if type(obj).__name__ == "DataFrame":
        print(name, obj.shape, list(obj.columns)[:25])

print("\n=== STATE INDEX ARRAYS ===")
for name, obj in sorted(list(globals().items())):
    if isinstance(obj, np.ndarray) and obj.ndim == 1 and len(obj) == 4035:
        print(name, obj.dtype, repr(obj[:4]))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python 173; forward 13 — abandon current posterior direction/gain approach\n"
            "- Gains (-40,-20) gave insulin x2 -2.047, half .791; NE x2 -.085, half .017 "
            "versus original -2.148, .819, .471, -.115.\n"
            "- Both signs flatten NE effects; the pathway is not calibrated causally. All edits reverted.\n"
            "- Next: investigate direct value-head response constraints using dose-perturbed recurrent features "
            "and factual-preservation constraints.\n")
```