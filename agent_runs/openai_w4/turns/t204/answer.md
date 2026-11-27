The saved insulin connection is not reflected in the CPU weight dictionary, so I’ll synchronize that before any local test. I’ll then compare the existing norepinephrine decoders and gate margins. A decoder should respond to half/double doses on held-out stays while remaining quiet when norepinephrine is absent.

```python
from pathlib import Path
import numpy as np, torch, inspect

rW = {n: torch.as_tensor(arr177(load(n)).copy(), dtype=torch.float32) for n in NAMES}
backup_before179 = {n: arr177(load(n)).copy() for n in edited()}

print("NE DECODER CANDIDATES")
print(raw_table87.loc[raw_table87["drug"] == "norepinephrine"].to_string(index=False))
print("\nGATE MODELS")
for drug179, v179 in gate_models88.items():
    print(drug179, type(v179).__name__)
    if isinstance(v179, dict):
        for k179, x179 in v179.items():
            print(k179, (getattr(x179, "shape", None), arr177(x179).reshape(-1)[:4].tolist())
                  if hasattr(x179, "shape") else str(x179)[:180])
    elif isinstance(v179, (tuple, list)):
        print([(getattr(x, "shape", None), str(x)[:100]) for x in v179])
    else:
        print(str(v179)[:250])
print("NE GATE MARGINS",
      [x for x in gate_rows88 if x.get("drug") == "norepinephrine"])

print("\nTARGET/CF INTERFACES")
for k179 in ["labels82", "eps_lookup82", "raw_cf87", "cfdr82",
             "raw_candidates87", "directions118", "probe_data95"]:
    v179 = globals().get(k179)
    if isinstance(v179, dict):
        print(k179, "keys", list(v179)[:8])
        if k179 in ["labels82", "eps_lookup82"]:
            x179 = v179["norepinephrine"]
            print("NE", type(x179).__name__, getattr(x179, "shape", None),
                  list(x179)[:10] if isinstance(x179, dict) else str(x179)[:200])
        elif k179 == "probe_data95":
            for kk179, xx179 in v179.items():
                print(kk179, getattr(xx179, "shape", type(xx179).__name__))
for k179 in ["decoder94", "importance94", "latent_axis104"]:
    x179 = arr177(globals()[k179])
    print(k179, x179.shape, "last", x179.reshape(-1)[-4:].tolist())
print("OTHER RESPONSE OBJECTS",
      " ".join(sorted(k for k in list(globals())
                      if any(str(n) in k for n in [118, 120, 160, 161, 166])
                      and not k.startswith("_"))))

with Path("/work/notes.md").open("a") as f179:
    f179.write("\n\n## Run 179\n"
               "Synchronized the CPU weight dictionary to submitted edits; previous CPU B_in "
               "was stale. Backed up submitted edits. Inspected NE decoder/gate candidates "
               "before selecting a held-out dose-sensitive channel. No saved weights changed.\n")
```