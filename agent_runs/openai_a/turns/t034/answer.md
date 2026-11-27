The retained data include matched GPU states and factual predictions, so we can test the head without rerunning the whole model. I’ll check the eight register fields, query type, and latent contribution, and measure errors separately by reading type. Systematic errors would indicate an encoding mistake; small, irregular differences could come from the GPU’s four latent draws.

```python
import dis
from collections import defaultdict
import numpy as np
import torch

print("=== HEAD FUNCTION SETTINGS ===")
print("defaults:", make_head_features.__defaults__)
print("locals:", make_head_features.__code__.co_varnames)

_groups = defaultdict(list)
_line = None
for _ins in dis.get_instructions(make_head_features):
    if _ins.starts_line is not None:
        _line = _ins.starts_line
    if _ins.opname in {
        "RESUME", "PRECALL", "CACHE", "NOP", "PUSH_NULL", "EXTENDED_ARG"
    }:
        continue
    _groups[_line].append(_ins.opname + ":" + _ins.argrepr)
print("\n=== COMPACT FUNCTION LOGIC ===")
_text = "\n".join(f"{_ln}: {' | '.join(_ops)}"
                  for _ln, _ops in _groups.items())
print(_text[:6200])

print("\n=== MATCHING DATA ===")
print("state stay IDs:",
      [(k, v["stay_id"][:2].tolist()) for k, v in state_files.items()])
print("factual columns:", factual_baseline.columns.tolist())
print(factual_baseline.head(2).to_string(index=False))
print("reference columns:", reference.columns.tolist())

print("\n=== DEFAULT HEAD OUTPUT ===")
_head_result = make_head_features(
    head_meta_tune, state_files[3],
    gap_style="seconds_div10", register_style="interleaved", status=0
)
if isinstance(_head_result, dict):
    print({k: getattr(v, "shape", type(v).__name__)
           for k, v in _head_result.items()})
elif isinstance(_head_result, (tuple, list)):
    print([(i, getattr(v, "shape", type(v).__name__))
           for i, v in enumerate(_head_result)])
else:
    print(type(_head_result).__name__, getattr(_head_result, "shape", None))
print("Current edits:", edited())
```