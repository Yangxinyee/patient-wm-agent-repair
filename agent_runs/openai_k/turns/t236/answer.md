The last completed run validates drug-bundle encoding, but not the readout used to calibrate routing directions. I’ll recover the architecture, record-derived targets, and official results before making another edit. Local calibration is trustworthy only if its predictions reproduce the official forward tool.

```python
from pathlib import Path
import numpy as np, pandas as pd, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== OFFICIAL RESULTS ===")
for k214 in [1, 2, 6, 10, 11, 12, 13, 14]:
    folder214 = Path(f"/work/fwd_{k214}")
    for p214 in sorted(folder214.glob("*.csv")):
        df214 = pd.read_csv(p214)
        print(k214, p214.name, "shape", df214.shape, "columns", list(df214.columns))
        if "dose_edit" in p214.name:
            print(df214.head(4).to_string(index=False))
        else:
            print("numeric means",
                  df214.select_dtypes(include="number").mean().round(6).to_dict())
    for p214 in sorted(folder214.glob("*.json")):
        print(k214, p214.name, p214.read_text()[:1800])

print("\n=== EARLY RECORD IDENTIFICATION EVIDENCE ===")
notes214 = Path("/work/notes.md").read_text().splitlines()
keys214 = ["causal", "instrument", "random", "impulse", "target", "confound",
           "identif", "true effect"]
matched214 = [i214 for i214, line214 in enumerate(notes214)
              if any(key214 in line214.lower() for key214 in keys214)]
for i214 in matched214[:40]:
    print(f"{i214}: {notes214[i214]}")

print("\n=== REPLAY HELPERS ===")
for name214, obj214 in list(globals().items()):
    if callable(obj214) and hasattr(obj214, "__code__"):
        if any(s214 in name214.lower() for s214 in ["head", "query", "causal", "effect"]):
            print(name214,
                  obj214.__code__.co_varnames[:obj214.__code__.co_argcount],
                  "uses", obj214.__code__.co_names)

with open("/work/notes.md", "a") as f214:
    f214.write("\n## Run 214: recover authoritative architecture and targets\n"
               "- Drug-bundle posterior encoding is validated; prior local value-head calibration is not.\n"
               "- Recovering architecture, official results, and record identification evidence before further edits.\n"
               "- No weights changed. Python runs: 214; forward passes: 14.\n")
```