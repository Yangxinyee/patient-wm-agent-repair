The recovered record estimates support a negative insulin effect on next-hour glucose and a positive norepinephrine effect on next-hour MAP. They do not explain the rollout asymmetry. I’ll inspect the installed channels and earlier GPU results before choosing another weight change; this will distinguish an encoding defect from insufficient response strength.

```python
from pathlib import Path
import numpy as np, torch, pandas as pd, inspect, re

print("=== SMALL DESIGN OBJECTS ===")
for _k in ["private_pairs", "reserved", "sensor_neurons", "_pair_for_control",
           "r249_sensor_controls", "_gate_table"]:
    if _k in globals():
        print(_k, globals()[_k])

print("\n=== COMPACT GLOBAL INVENTORY ===")
_out253 = []
for _k, _v in sorted(dict(globals()).items()):
    if _k.startswith("_") or inspect.isfunction(_v) or inspect.ismodule(_v):
        continue
    if isinstance(_v, (torch.Tensor, np.ndarray)):
        _out253.append(f"{_k}:{tuple(_v.shape)}")
    elif isinstance(_v, pd.DataFrame):
        _out253.append(f"{_k}:DF{_v.shape}")
    elif isinstance(_v, dict):
        _out253.append(f"{_k}:dict({len(_v)}) keys={list(_v)[:5]}")
    elif isinstance(_v, (list, tuple)):
        _out253.append(f"{_k}:{type(_v).__name__}({len(_v)})")
print("\n".join(_out253)[:3700])

print("\n=== FORWARD CSV SCHEMAS ===")
for _folder in ["/work/fwd_1", "/work/fwd_17", "/work/fwd_21", "/work/fwd_23"]:
    for _p in sorted(Path(_folder).glob("*.csv")):
        _df = pd.read_csv(_p)
        print(str(_p), _df.shape, list(_df.columns), _df.head(1).to_dict("records"))

print("\n=== EARLY TRANSPORT NOTES ===")
_ss253 = re.split(r"(?m)(?=^## )", Path("/work/notes.md").read_text())
_sel253 = [s for s in _ss253 if re.search(
    r"(reserved|router|routing|action encoder|end atom|private|clean transport)", s, re.I)]
print("\n".join(s[:1100] for s in _sel253[:5])[:4200])

W253 = {n: load(n) for n in NAMES}
print("\n=== EMBEDDING TABLE SHAPES ===")
print({n: tuple(W253[n].shape) for n in NAMES
       if n.startswith("embed.") or n.startswith("encode_action.")})
```