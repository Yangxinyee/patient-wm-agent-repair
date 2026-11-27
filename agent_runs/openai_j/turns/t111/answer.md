The recovered estimates support an immediate glucose reduction from insulin and MAP increase from norepinephrine, but insulin’s persistence is weakly identified. I will keep that uncertainty separate from the reconstruction problem and inspect the cached head inputs. A usable reconstruction should reproduce GPU predictions without systematic sensor-dependent bias.

```python
import numpy as np
import pandas as pd
from pathlib import Path

print("=== RELEVANT TENSOR SHAPES ===")
for name in NAMES:
    if any(s in name for s in [
        "last_numeric", "last_status", "query_norm", "type_emb",
        "z_head", "z_prior", "attr_heads.value", "embed_transition.value"
    ]):
        print(name, SHAPES[name])

print("\n=== CACHED DATAFRAMES ===")
for name, value in list(globals().items()):
    if isinstance(value, pd.DataFrame):
        print(name, value.shape, list(value.columns)[:22])

print("\n=== RECONSTRUCTION FUNCTIONS ===")
for name, value in list(globals().items()):
    if callable(value) and hasattr(value, "__code__"):
        if any(s in name.lower() for s in [
            "repair", "readout", "head", "numeric", "context",
            "bundle", "layout", "predict", "reconstruct"
        ]):
            c = value.__code__
            print(name, "args:", c.co_varnames[:c.co_argcount],
                  "refs:", c.co_names)

print("\n=== FORWARD ARTIFACTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])

print("\n=== VOCAB CONFIGURATION ===")
print("keys:", list(VOCAB))
print("config:", VOCAB.get("config"))
print("codes:", VOCAB.get("codes"))

notes = Path("/work/notes.md").read_text()
print("\n=== RECONSTRUCTION AND PROBE NOTES ===")
a = notes.find("## Head reconstruction setup")
b = notes.find("## Run 53")
print(notes[a:b][:4300])
```