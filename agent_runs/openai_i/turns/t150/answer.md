The inspection run timed out and reset the kernel. No edits were active, so the model remains unchanged, but the cached helpers are lost. I’ll rebuild from the saved files with bounded output. The next checkpoint is an exact state-update reconstruction; if it cannot reproduce exposed states, I will not use it to choose an edit.

```python
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open

torch.set_grad_enabled(False)
torch.set_num_threads(8)

print("=== PROVIDED API AFTER RESET ===")
for name in ["NAMES", "SHAPES", "VOCAB", "load", "load_original",
             "save_edit", "drop_edit", "edited"]:
    print(name, name in globals())

v135 = json.loads(Path("/model/vocab.json").read_text())
print("\n=== VOCAB STRUCTURE ===")
print("keys:", list(v135))
print("config:", v135.get("config"))
print("codes:", v135.get("codes"))
for key in ["normalisation", "normalization", "event_kinds", "categorical"]:
    if key in v135:
        print(key, str(v135[key])[:900])

print("\n=== DATA DOCUMENTATION ===")
print(Path("/data/README.md").read_text()[:2600])
for split in ["train", "tuning"]:
    tiny = pd.read_csv(f"/data/{split}.csv", nrows=3)
    print(split, tiny.to_dict("records"))

print("\n=== SAVED FORWARD FILES ===")
for directory in sorted(Path("/work").glob("fwd_*")):
    print(directory.name, [p.name for p in directory.iterdir()])

print("\n=== RETAINED RECORD-EFFECT EVIDENCE ===")
notes135 = Path("/work/notes.md").read_text()
sections135 = re.split(r"(?=^## )", notes135, flags=re.M)
for section in sections135:
    if section.startswith("## Lagged-effect investigation"):
        print(section[:2300])
        break

print("\n=== WEIGHT ACCESS ===")
with safe_open("/model/model.safetensors", framework="pt", device="cpu") as sf135:
    names135 = list(sf135.keys())
    print("tensor count:", len(names135))
    print("gap tensors:", [n for n in names135 if "gap" in n.lower()])
    print("attention-related tensors:",
          [(n, tuple(sf135.get_tensor(n).shape))
           for n in names135
           if n.startswith("core.prelude") or n.startswith("core.age")])

if callable(globals().get("edited")):
    print("current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 135 — kernel reset recovery\n"
        "- Python 134 timed out and reset all cached variables; no edits were active.\n"
        "- Reloading bounded file metadata and retained causal evidence.\n"
        "- Rebuilding the recurrent calculation requires exposed-state validation "
        "before any response-based weight edit.\n"
    )
```