The records support opposite-signed drug effects, but the early estimates vary across measurement channels and do not yet specify the full rollout response. The output was truncated before the retained fits appeared. I’ll retrieve those objects and the later robustness checks in a smaller output; agreement across those checks would justify a targeted edit rather than a raw dose-response correction.

```python
from pathlib import Path
import numpy as np

_n60 = Path("/work/notes.md").read_text()
print("=== LATER CAUSAL CHECKS ===")
_p60 = _n60.find("## Run 6")
_q60 = _n60.find("## Run 25")
print(_n60[_p60:_q60][:6200] if _p60 >= 0 else _n60[5000:11200])

print("\n=== CANDIDATE OBJECT INVENTORY ===")
for _nm60 in sorted(list(globals())):
    if _nm60.startswith(("rr50", "rr51", "rr56", "rr57", "rr58", "r58")):
        _vv60 = globals()[_nm60]
        if hasattr(_vv60, "shape"):
            print(_nm60, type(_vv60).__name__, tuple(_vv60.shape))
        elif isinstance(_vv60, dict):
            print(_nm60, "dict", list(_vv60)[:10])
        elif isinstance(_vv60, (list, tuple)):
            print(_nm60, type(_vv60).__name__, len(_vv60))
        elif callable(_vv60):
            print(_nm60, "callable")
        else:
            print(_nm60, repr(_vv60)[:100])

print("\n=== POSTERIOR / ACTION TENSOR SHAPES ===")
for _nm60 in NAMES:
    if any(_s60 in _nm60 for _s60 in
           ("z_post", "z_in", "z_head", "encode_action", "embed.numeric")):
        print(_nm60, SHAPES[_nm60])
with open("/work/notes.md", "a") as _f60:
    _f60.write("\n## Run 60\n"
               "Run 59 output was truncated. Recovering later causal checks and candidate dimensions "
               "without repeating architecture. No weights changed. "
               "Counts: 60 Python runs, 6 forward passes.\n")
```