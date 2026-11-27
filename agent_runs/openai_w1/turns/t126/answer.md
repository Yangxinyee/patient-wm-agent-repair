The value head is a two-layer network, so a localized readout edit may be possible. The earlier regressions estimate roughly −2.2 glucose units per insulin-rate unit and +3.3 MAP units per norepinephrine-rate unit at the next reading, but rollout effects also depend on persistence. I’ll recover those dynamics and identify the validated reconstruction instead of repeating unsuccessful searches.

```python
from pathlib import Path
import re, numpy as np, torch

_text = Path("/work/notes.md").read_text()
_sections = re.split(r"(?m)(?=^## )", _text)
print("=== TARGETED NOTEBOOK RECOVERY ===")
_budget = 6100
for _s in _sections:
    _title = _s.splitlines()[0] if _s.splitlines() else ""
    if any(k in _s.lower() for k in
           ("deconvol", "kernel", "trace decoder", "head reconstruction", "exact",
            "forward 4", "forward 5", "forward 6")) and not any(
                f"Run {i}" in _title for i in range(82, 116)):
        _chunk = _s[:min(950, _budget)]
        print(_chunk)
        _budget -= len(_chunk)
        if _budget <= 0:
            break

print("\n=== VALIDATED HELPER DEPENDENCIES ===")
for _n in ("r88_local_atoms", "r88_local_score", "r86_encode_observations",
           "r89_encode_actions", "r95_base_atom", "r95_correct_atom", "rr_atom",
           "rr_encode_atoms112"):
    _fn = globals().get(_n)
    if hasattr(_fn, "__code__"):
        print(_n, "globals:", _fn.__code__.co_names,
              "constants:", [v for v in _fn.__code__.co_consts
                             if isinstance(v, (str, int, float, tuple))][:16])

print("\n=== FRESH ARRAY/TABLE INVENTORY ===")
for _n, _v in sorted(list(globals().items())):
    if _n.startswith("rr_"):
        if isinstance(_v, (np.ndarray, torch.Tensor)):
            print(_n, tuple(_v.shape))
        elif hasattr(_v, "columns") and hasattr(_v, "shape"):
            print(_n, "table", _v.shape, list(_v.columns)[:10])
        elif isinstance(_v, dict):
            print(_n, "dict", len(_v), list(_v)[:4])

with open("/work/notes.md", "a") as _f:
    _f.write("\n## Run 115\nLocalized value-head editing is structurally possible, but dynamic effect "
             "and exact feature calibration are required. Recovering prior validated helpers rather "
             "than repeating broad reconstruction searches. No edits. "
             "Counts: 115 Python runs, 6 forward passes.\n")
```
