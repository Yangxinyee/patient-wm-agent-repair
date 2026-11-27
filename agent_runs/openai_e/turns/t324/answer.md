```python
from pathlib import Path
import numpy as np
import pandas as pd

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\nCURRENT EDITS", edited())
print("\nSAVED CANDIDATE CONTRACTS")
for fn in ["joint236.npz", "joint268.npz", "joint274.npz",
           "insulin_candidate63.npz", "ne_candidate151.npz"]:
    with np.load("/work/" + fn, allow_pickle=False) as a:
        print(fn, {k: (a[k].item() if a[k].size == 1 else
                       {"shape": a[k].shape, "norm": float(np.linalg.norm(a[k]))})
                   for k in a.files})

print("\nPAIRED FACTUAL RESULTS")
for k in [33, 34, 36, 37]:
    folder = Path(f"/work/fwd_{k}")
    paths = [folder / f"factual_{m}.csv" for m in ["original", "edited"]]
    if all(p.exists() for p in paths):
        a, b = [pd.read_csv(p) for p in paths]
        print("forward", k, "columns", list(a.columns))
        print("original first row", a.head(1).to_dict("records"))
        print("edited first row", b.head(1).to_dict("records"))
        nllcols = [c for c in a if "nll" in c.lower()]
        for c in nllcols:
            print(c, "original", float(a[c].mean()), "edited", float(b[c].mean()),
                  "delta", float(b[c].mean() - a[c].mean()))
        for c in a:
            if "stay" in c.lower():
                print("cohort", c, sorted(a[c].unique().tolist()))
                break

print("\nRETAINED VARIABLE NAMES")
print(" ".join(sorted(k for k in globals()
                      if not k.startswith("_") and k not in
                      ["NAMES", "SHAPES", "VOCAB", "In", "Out"])))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 275: cross-cohort audit of joint274\n"
            "Forward37 reported factual delta about .00206 on 100 tuning stays, "
            "below .005. Recovering exact forward36/37 evidence, architecture "
            "and saved candidate contracts before acceptance or further edits. "
            "Weights unchanged. Python runs: 275; forwards: 37.\n")
```