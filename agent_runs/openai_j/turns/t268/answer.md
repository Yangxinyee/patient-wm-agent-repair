The nonlinear candidates improve norepinephrine doubling only slightly on held-out stays, so I won’t install them yet. I’ll recover the architecture and causal evidence next. A useful correction must improve doubling without sacrificing halving or factual fit.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import inspect

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\nEARLY NOTEBOOK")
print(Path("/work/notes.md").read_text()[:14000])

print("\nRELEVANT KERNEL OBJECTS")
_g238 = dict(globals())
for _k238 in sorted(_g238):
    if (any(s in _k238.lower() for s in
            ["233", "234", "235", "236", "237", "head177", "map192",
             "kernel_target", "causal", "iv_results", "dose_feature"])
        and not _k238.startswith("_")):
        _v238 = _g238[_k238]
        if callable(_v238):
            try:
                print(_k238, inspect.signature(_v238))
                print(inspect.getsource(_v238)[:7000])
            except Exception:
                print(_k238, type(_v238).__name__)
        elif isinstance(_v238, dict):
            print(_k238, "dict", len(_v238), "keys:", list(_v238)[:20])
        elif hasattr(_v238, "shape"):
            print(_k238, type(_v238).__name__, _v238.shape)
        elif isinstance(_v238, (list, tuple)):
            print(_k238, type(_v238).__name__, len(_v238), str(_v238[:8])[:1200])
        else:
            print(_k238, str(_v238)[:1000])

for _f238 in ["kernel_targets.csv", "dynamic_iv.csv", "sensor_iv.csv",
              "dml_results.csv", "proxy_iv.csv"]:
    print("\n", _f238)
    _df238 = pd.read_csv("/work/" + _f238)
    print("shape:", _df238.shape)
    print(_df238.head(16).to_string(index=False))

with open("/work/notes.md", "a") as _n238:
    _n238.write(
        "\n## Run 238: restart recovery\n"
        "- Nonlinear MAP candidates show only modest held-out doubling gains; none installed.\n"
        "- Recovering architecture and causal evidence before further edits.\n"
        "- GPU-validated gain-8 four-tensor edit remains active.\n"
    )
```