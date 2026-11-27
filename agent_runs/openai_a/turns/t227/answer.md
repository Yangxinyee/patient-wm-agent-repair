The saved screens suggest that stronger regularization may permit a larger correction at a similar factual cost. I will install the regularization-0.3, scale-6 variant as a reversible trial. GPU results must show both acceptable factual loss and a better insulin-doubling response; cached results alone are insufficient.

```python
import numpy as np
import torch
from pathlib import Path
import inspect

assert "finger_scale3_checkpoint" in globals()
alt_contrast_config = {
    "regularization": 0.3,
    "scale": 6.0,
    "gate_strength": 30.0,
}
_alt = r3_candidates[(0.3, 1.0)]
_alt_weights = make_selective_contrast_graft(
    _alt["w"], _alt["b"],
    alt_contrast_config["scale"],
    gate_strength=alt_contrast_config["gate_strength"],
)
assert set(_alt_weights) == set(finger_scale3_checkpoint)
for _name, _value in _alt_weights.items():
    _tensor = (
        _value.detach().clone().float()
        if isinstance(_value, torch.Tensor)
        else torch.from_numpy(np.asarray(_value, dtype=np.float32).copy())
    )
    assert tuple(_tensor.shape) == tuple(SHAPES[_name])
    assert torch.isfinite(_tensor).all()
    save_edit(_name, _tensor)

print("INSTALLED REVERSIBLE TRIAL:", alt_contrast_config)
print("EDITED:", edited())
print("CACHE HELPER:", inspect.signature(selective_cache))
print("AVAILABLE CACHE DICTIONARIES:")
for _name, _obj in list(globals().items()):
    if _name.startswith("_") or not isinstance(_obj, dict):
        continue
    if {"x", "bins", "logits"}.issubset(_obj):
        print(_name, {k: getattr(v, "shape", None)
                      for k, v in _obj.items()})
    elif "cache" in _name.lower():
        print(_name, "keys=", list(_obj)[:8])

with Path("/work/notes.md").open("a") as _f:
    _f.write(
        "\n## Stronger-regularization trial; Python 200, forwards 18/50\n"
        "- Installed selective r3 lambda=0.3, scale=6, gate strength=30.\n"
        "- Finger scale3 and scale2 validated checkpoints remain preserved.\n"
        "- Candidate is unaccepted pending paired GPU factual/rollout checks.\n"
    )
```