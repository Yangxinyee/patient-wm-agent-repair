The state packing is correct, so it does not explain the reconstruction error. The bin mismatches and different time scalings still need checking. I’ll recover the architecture and cached analyses before editing; a reliable reconstruction should agree with the forward tool within latent-sampling noise.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())
print("=== EARLY NOTEBOOK: CAUSAL EVIDENCE ===")
print(Path("/work/notes.md").read_text()[:11500])

print("=== PERSISTENT DATA / FUNCTIONS ===")
_r34_globals = list(globals().items())
for _r34_name, _r34_obj in sorted(_r34_globals):
    if _r34_name.startswith("_"):
        continue
    if isinstance(_r34_obj, pd.DataFrame):
        print(_r34_name, "DataFrame", _r34_obj.shape,
              "columns:", list(_r34_obj.columns))
    elif isinstance(_r34_obj, (np.ndarray, torch.Tensor)):
        print(_r34_name, type(_r34_obj).__name__, tuple(_r34_obj.shape))
    elif isinstance(_r34_obj, dict):
        _r34_keys = list(_r34_obj.keys())
        print(_r34_name, "dict", len(_r34_keys), "keys:", _r34_keys[:35])
    elif callable(_r34_obj) and hasattr(_r34_obj, "__code__"):
        print(_r34_name, "function",
              _r34_obj.__code__.co_varnames[:_r34_obj.__code__.co_argcount])

print("=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as _r34_f:
    _r34_f.write(
        "\n## Run 34: restart recovery\n"
        "Run 33 verified exact state packing. Last-bin boundaries disagree in "
        "163 cached rows; prior and reading-view gap scalings differ. "
        "Recovering architecture and causal caches before any weight edit.\n"
    )
```