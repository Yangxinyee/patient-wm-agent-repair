```python
from pathlib import Path
import numpy as np
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== EARLY NOTEBOOK: IDENTIFICATION AND CHANNEL DESIGN ===")
_notes258 = Path("/work/notes.md").read_text()
print(_notes258[:18000])

print("\n=== RELEVANT TENSOR SHAPES ===")
for _n258 in NAMES:
    if any(s in _n258 for s in (
        "core.flow", "core.delta", "core.slots", "core.b_in", "core.c_out",
        "out_norm", "value_bin", "encode_action", "last_numeric"
    )):
        print(_n258, SHAPES[_n258])

print("\n=== RECOVERABLE KERNEL OBJECTS ===")
_obj258 = []
for _n258, _v258 in list(globals().items()):
    if _n258.startswith("_") or _n258 in {"NAMES", "SHAPES", "VOCAB"}:
        continue
    if isinstance(_v258, (np.ndarray, torch.Tensor)):
        _obj258.append(f"{_n258}: {type(_v258).__name__} {_v258.shape}")
    elif isinstance(_v258, dict):
        _obj258.append(f"{_n258}: dict[{len(_v258)}] keys={list(_v258)[:8]}")
    elif callable(_v258):
        if getattr(_v258, "__module__", "") == "__main__":
            _obj258.append(f"{_n258}: callable")
    elif type(_v258).__name__ in {"DataFrame", "Series"}:
        _obj258.append(f"{_n258}: {type(_v258).__name__} {_v258.shape}")
    elif isinstance(_v258, (list, tuple)) and len(_v258) < 30:
        _obj258.append(f"{_n258}: {str(_v258)[:180]}")
print("\n".join(_obj258)[:22000])

with open("/work/notes.md", "a") as _f258:
    _f258.write(
        "\n## Run 258: persistence-trial preparation\n"
        "- Hourly paired effects understate record-derived insulin responses, especially halving.\n"
        "- Rapid private-channel decay is a candidate mechanism, not an established explanation.\n"
        "- Recovering exact architecture, causal estimates, and current edit layout before a reversible trial.\n"
        "- No weights changed.\n"
    )
```