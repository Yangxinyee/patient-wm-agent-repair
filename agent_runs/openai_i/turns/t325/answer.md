The diagnostic failed because it mixed NumPy arrays and Torch tensors; it provides no evidence for or against the gain-0.8 edit. I’ll recover the latest validation results and record-based targets using consistent types. The edit remains provisional until its factual loss satisfies the limit and its dose responses improve.

```python
import os, glob, json, re
import numpy as np
import pandas as pd
import torch

def as_numpy282(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("ARCHITECTURE")
print(open("/model/ARCHITECTURE.md").read())

notes282 = open("/work/notes.md").read()
print("\nRECORD-BASED TARGET EVIDENCE")
evidence282 = [
    line for line in notes282.splitlines()
    if re.search(r"target|causal|identif|true effect|dose.response|ground.truth", line, re.I)
]
print("\n".join(evidence282[-55:]))

print("\nLATEST FORWARD FILES")
for k282 in [28, 29, 30, 31]:
    directory282 = f"/work/fwd_{k282}"
    if not os.path.isdir(directory282):
        continue
    print(directory282, os.listdir(directory282))
    for path282 in sorted(glob.glob(directory282 + "/*.csv")):
        df282 = pd.read_csv(path282)
        print(os.path.basename(path282), "shape", df282.shape,
              "columns", df282.columns.tolist())
        print(df282.head(2).to_dict("records"))
        nllcols282 = [c for c in df282.columns if "nll" in c.lower()]
        if nllcols282:
            print("mean NLL columns",
                  df282[nllcols282].apply(pd.to_numeric, errors="coerce").mean().to_dict())
        if "dose_edit" in path282:
            labelcols282 = [
                c for c in df282.columns
                if any(s in c.lower() for s in ["edit", "intervention", "scenario"])
                and not pd.api.types.is_numeric_dtype(df282[c])
            ]
            numeric282 = df282.select_dtypes(include=np.number).columns.tolist()
            if labelcols282:
                print(df282.groupby(labelcols282)[numeric282].mean().to_string())

print("\nCHECKPOINT CONTENTS")
for path282 in ["/work/neutral209.npz", "/work/corrected269.npz", "/work/gain278.npz"]:
    with np.load(path282, allow_pickle=False) as cp282:
        print(os.path.basename(path282),
              {key: cp282[key].shape for key in cp282.files})

print("\nCURRENT EDITS", edited())
print("\nRETAINED HELPERS / TARGET VARIABLES")
names282 = [
    key for key in list(globals())
    if not key.startswith("_")
    and re.search(r"target|causal|effect|replay|compens|direction|stay|install|route", key, re.I)
]
for key282 in names282[-90:]:
    value282 = globals()[key282]
    if callable(value282):
        print(key282, "callable")
    elif isinstance(value282, (np.ndarray, torch.Tensor, pd.DataFrame)):
        print(key282, type(value282).__name__, tuple(value282.shape))
    elif isinstance(value282, dict):
        print(key282, "dict keys", list(value282)[:16])
    elif isinstance(value282, (int, float, str, list, tuple)):
        text282 = repr(value282)
        print(key282, text282[:240])

with open("/work/notes.md", "a") as log282:
    log282.write(
        "\n## Python 282 — restart recovery after mixed-type diagnostic failure\n"
        "- Gain-0.8 corrected269 remains installed and provisional.\n"
        "- Recovering architecture, recorded causal targets, and latest GPU validation files.\n"
        "- Explicit NumPy conversion avoids ndarray/Tensor subtraction errors; no weights changed.\n"
    )
```