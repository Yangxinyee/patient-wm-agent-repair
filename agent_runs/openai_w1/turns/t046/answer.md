The inventory failed because it mistook NumPy’s `shape` function for an array’s dimensions. This tells us nothing new about drug effects, and no weights changed. I’ll use explicit type checks to recover the cached analyses safely; success means identifying the existing panels and regression helpers without rebuilding them.

```python
import types
import numpy as np
import pandas as pd
import torch
from pathlib import Path

_items = list(globals().items())
_fn_lines, _data_lines = [], []
for _name, _obj in sorted(_items):
    if _name.startswith("_") or _name in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_obj, types.ModuleType):
        continue
    if isinstance(_obj, types.FunctionType):
        if _obj.__module__ == "__main__":
            _args = _obj.__code__.co_varnames[:_obj.__code__.co_argcount]
            _fn_lines.append(_name + "(" + ", ".join(_args) + ")")
    elif isinstance(_obj, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        _data_lines.append(f"{_name}: {type(_obj).__name__} {tuple(_obj.shape)}")
    elif isinstance(_obj, dict):
        _keys = repr(list(_obj.keys())[:6])[:100]
        _data_lines.append(f"{_name}: dict[{len(_obj)}] {_keys}")
    elif isinstance(_obj, (list, tuple)):
        _data_lines.append(f"{_name}: {type(_obj).__name__}[{len(_obj)}]")
    elif isinstance(_obj, (str, int, float, bool, np.number)):
        _data_lines.append(f"{_name}: {repr(_obj)[:65]}")

print("=== FUNCTIONS ===")
print("\n".join(_fn_lines)[:2200])
print("\n=== DATA AND SETTINGS ===")
print("\n".join(_data_lines)[:4800])

print("\n=== VOCAB ===")
print("keys:", list(VOCAB.keys()))
for _key in VOCAB:
    if any(s in str(_key).lower() for s in ("config", "norm", "codes")):
        print(_key, repr(VOCAB[_key])[:1200])

print("\n=== RELEVANT TENSORS ===")
for _name in NAMES:
    if any(s in _name for s in ("z_post", "z_prior", "z_head", "last_numeric",
                                "query_norm", "type_emb", "encode_action")):
        print(_name, SHAPES[_name])

with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Restart, runs 36–39\n"
             "Run 36 reconstruction failed on tuple/dict indexing; posterior reconstruction "
             "remained unreliable outside action bundles. Run 37 read exact architecture. "
             "Run 38 inventory failed on a module's callable shape attribute. "
             "Run 39 uses explicit type checks to recover persistent analyses. "
             "No weights changed; 39 Python runs, 6 forward passes.\n")
```