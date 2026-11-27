```python
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED ANALYSIS OBJECTS ===")
for _name, _obj in sorted(list(globals().items())):
    if _name.startswith("_") or _name in {
        "NAMES", "SHAPES", "VOCAB", "In", "Out", "get_ipython",
        "load", "load_original", "save_edit", "drop_edit", "edited"
    }:
        continue
    if isinstance(_obj, pd.DataFrame):
        print(_name, "DataFrame", _obj.shape, "columns:", list(_obj.columns))
    elif isinstance(_obj, (np.ndarray, torch.Tensor)):
        print(_name, type(_obj).__name__, tuple(_obj.shape))
    elif isinstance(_obj, dict):
        print(_name, "dict keys:", list(_obj.keys())[:35])
    elif hasattr(_obj, "predict") or hasattr(_obj, "fit"):
        print(_name, type(_obj).__name__)

print("\n=== ACTION, STATE, AND OUTPUT TENSORS ===")
for _name in NAMES:
    if any(_term in _name.lower() for _term in
           ("action", "dose", "drug", "head", "out", "embed", "slot", "prior", "post")):
        print(_name, SHAPES[_name])

print("\n=== VOCABULARY TOP LEVEL ===")
for _key, _value in VOCAB.items():
    _text = repr(_value)
    print(_key, _text if len(_text) <= 2400 else _text[:2400] + " ...")

print("\n=== FORWARD ARTIFACTS ===")
for _folder in ("/work/fwd_1", "/work/fwd_2"):
    print(_folder, sorted(os.listdir(_folder)))
print("Current edits:", edited())

with open("/work/notes.md", "a") as _f:
    _f.write(
        "\n## Run 7: restart inspection\n"
        "- Re-read architecture and schema and inventoried retained analysis objects before choosing an edit.\n"
        "- No new causal conclusion or weight edit in this run.\n"
        "- Python runs: 7; forward passes: 2.\n"
    )
```