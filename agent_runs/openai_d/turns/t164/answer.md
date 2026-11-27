The norepinephrine edit moved doubling from **−0.498 to +2.351**, supporting the intended pathway, but halving remains too weak at **−0.893**. Insulin stayed near **−10.26/+5.02**. Simply amplifying both would risk factual fit. I’ll inspect how the correction behaves on factual states and how its decoder was fitted; better separation of dose changes from ordinary state variation would justify further amplification.

```python
import re
import inspect
from pathlib import Path
import numpy as np
import torch

notes_text = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", notes_text)
for prefix, limit in [
    ("## Run 97", 1000),
    ("## Run 98", 1000),
    ("## Run 99", 1600),
    ("## Run 103", 900),
    ("## Run 104", 1800),
    ("## Run 105", 1000),
    ("## Run 106", 700),
]:
    matches = [s for s in sections if s.startswith(prefix)]
    if matches:
        print("\n" + matches[0][:limit])

print("\n=== HELPER SIGNATURES ===")
for name in ["rh_eval", "rh_register", "local104", "r124_context",
             "r129_queries", "response_arrays", "summary102"]:
    fn_here = globals().get(name)
    if callable(fn_here):
        print(name, inspect.signature(fn_here))
        print("  globals:", fn_here.__code__.co_names)
        print("  variables:", fn_here.__code__.co_varnames[:18])

print("\n=== EARLIER DECODER/CACHE NAMES ===")
selected_names = [
    k for k in globals()
    if re.search(r"(98|99|103|104|105|106)$", k)
       or k.startswith(("r103", "r104", "r105", "r106"))
]
for k in sorted(selected_names)[:65]:
    v = globals()[k]
    if isinstance(v, dict):
        print(k, "dict", list(v.keys())[:10])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    elif callable(v):
        print(k, "function")
    else:
        print(k, type(v).__name__, str(v)[:100])

print("\n=== FACTUAL CACHE STAYS ===")
print(head_meta[["stay_id", "event"]].groupby("event").agg(["count"]))
print("stay range:", head_meta.stay_id.min(), head_meta.stay_id.max())
print("NE feature keys:", r139_feature.keys())
print("NE feature factual SD:", np.std(r139_feature["pair"]))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Forward 17 / Run 143: actual NE response is partial\n"
        "Current four-head-tensor trial on tuning 2020000000..99: "
        "insulin x2 -10.255, half +5.021; NE x2 +2.351, half -0.893. "
        "NE doubling now has the record-supported sign, but both NE effects "
        "remain below provisional targets +4.746/-2.373. "
        "Factual delta +0.007284 on tuning 300..399 remains unacceptable "
        "as a standalone check. Investigating factual feature leakage and "
        "decoder fitting before amplification; no new edit installed.\n"
    )
```