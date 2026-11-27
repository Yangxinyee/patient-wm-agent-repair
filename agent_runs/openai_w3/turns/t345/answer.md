The four shifts changed only slightly, so removing this local context subspace did not explain the asymmetry. Doubled insulin is much improved, but the other responses remain too weak relative to the record-derived estimates. I’ll inspect the retained dose-carrier and checkpoint details before choosing another reversible edit; any candidate must pass both factual and rollout checks.

```python
import inspect
import numpy as np
import torch
from pathlib import Path

_r296_g = list(globals().items())
checkpoint296 = {k: load(k).clone() for k in edited()}

print("=== CHECKPOINT INVENTORY ===")
for _r296_k, _r296_v in _r296_g:
    if "checkpoint" in _r296_k.lower() and isinstance(_r296_v, dict):
        print(_r296_k, "entries", len(_r296_v),
              "first keys", list(_r296_v)[:3])

print("=== RUN294 RETAINED VARIABLES ===")
for _r296_k, _r296_v in _r296_g:
    if "294" not in _r296_k or _r296_k.startswith("_r295"):
        continue
    if isinstance(_r296_v, (np.ndarray, torch.Tensor)):
        _r296_a = (_r296_v.detach().cpu().numpy()
                   if isinstance(_r296_v, torch.Tensor) else _r296_v)
        print(_r296_k, _r296_a.shape,
              "head", _r296_a.ravel()[:6].tolist())
    elif isinstance(_r296_v, dict):
        print(_r296_k, "dict keys", list(_r296_v)[:12])
    elif isinstance(_r296_v, str):
        print(_r296_k, "string", len(_r296_v), _r296_v[:180])
    else:
        print(_r296_k, repr(_r296_v)[:180])

print("=== POLICY / CARRIER / CONTEXT INVENTORY ===")
_r296_items = [
    (k, v) for k, v in _r296_g
    if not k.startswith("_")
    and any(s in k.lower() for s in ("policy", "carrier", "ctx", "phys", "total_response"))
]
for _r296_k, _r296_v in _r296_items[:60]:
    if isinstance(_r296_v, (np.ndarray, torch.Tensor)):
        print(_r296_k, tuple(_r296_v.shape))
    elif inspect.isfunction(_r296_v):
        print(_r296_k, inspect.signature(_r296_v))
        try:
            _r296_src = inspect.getsource(_r296_v)
            print(_r296_src[:1300])
        except Exception:
            pass
    elif isinstance(_r296_v, dict):
        print(_r296_k, "dict keys", list(_r296_v)[:10])
    else:
        print(_r296_k, repr(_r296_v)[:130])

print("=== RETAINED IMPLEMENTATION STRINGS ===")
for _r296_k, _r296_v in _r296_g:
    if (isinstance(_r296_v, str) and "def " in _r296_v
        and any(s in _r296_v for s in ("carrier", "total_response", "policy"))):
        print(_r296_k, "length", len(_r296_v),
              "definitions", [s.strip() for s in _r296_v.splitlines()
                              if s.lstrip().startswith("def ")][:12])

print("=== SELECTED RECORD EVIDENCE ===")
_r296_lines = Path("/work/notes.md").read_text().splitlines()
_r296_evidence = [s for s in _r296_lines
                  if any(t in s.lower() for t in
                         ("identified", "identification", "instrument",
                          "physiolog", "true effect", "record-derived reference"))]
for _r296_s in _r296_evidence[:8]:
    print(_r296_s[:450])
print("=== CARRIER / POLICY HISTORY ===")
_r296_history = [s for s in _r296_lines
                 if any(t in s.lower() for t in
                        ("row31", "row 31", "row93", "row 93",
                         "carrier", "policy", "total-response"))]
for _r296_s in _r296_history[-14:]:
    print(_r296_s[:450])

print("Posterior first-layer shapes:",
      tuple(load("z_post.0.weight").shape),
      tuple(load("z_post.0.bias").shape))
with open("/work/notes.md", "a") as _r296_f:
    _r296_f.write(
        "\nForwards 33-34: centered context projection passed factual on "
        "100 tuning stays (1.64168 versus 1.63982; delta +.00186). "
        "Dose shifts (-12.318,+1.412,+.418,-1.203): negligible change "
        "from alpha=.01, arguing against the tested local feedback "
        "subspace as the main cause of asymmetry.\n"
        "Python run 296; forwards 34. Preserved current validated weights "
        "in checkpoint296; bounded recovery of carrier/policy implementation "
        "and record-only evidence. No weights changed.\n"
    )
```