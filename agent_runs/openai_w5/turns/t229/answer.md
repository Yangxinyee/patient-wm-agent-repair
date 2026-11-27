The repair uses four private state coordinates and eight reserved value-head neurons. The saved checkpoint makes further tests reversible. I’ll inspect how those neurons decode dose and gate each sensor; a nonlinear encoder response or a clipped gate could explain why doubling and halving behave differently.

```python
from pathlib import Path
import json, numpy as np, pandas as pd, torch

print("=== NOTEBOOK: TRANSPORT DESIGN ===")
print(Path("/work/notes.md").read_text()[12500:17400])

priv = [15, 21, 61, 66]
reserved = [7, 10, 64, 74, 107, 109, 124, 171]
A = W["attr_heads.value_bin.0.weight"]
B = W["attr_heads.value_bin.0.bias"]
C = W["attr_heads.value_bin.2.weight"]
print("\n=== RESERVED NEURONS ===")
for j in reserved:
    ordinary = A[j].clone()
    ordinary[priv] = 0
    ii = ordinary.abs().argsort(descending=True)[:4]
    print(j, "bias", round(float(B[j]), 5),
          "private", [round(float(x), 3) for x in A[j, priv]],
          "ordinary_norm", round(float(ordinary.norm()), 3),
          "largest_ordinary",
          [(int(i), round(float(A[j, i]), 3)) for i in ii],
          "output_norm", round(float(C[:, j].norm()), 5),
          "output_1_8_16", C[[1, 8, 16], j].tolist())

print("\n=== NUMERIC SCHEMA ===")
print(json.dumps(V["numeric"], indent=2)[:1900])
print("\n=== RECENT FORWARD SCHEMA ===")
for folder in ["fwd_15", "fwd_17", "fwd_18"]:
    p = Path("/work") / folder
    if (p / "summary.json").exists():
        print(folder, (p / "summary.json").read_text()[:900])
    for filename in ["factual_edited.csv", "dose_edit_edited.csv"]:
        if (p / filename).exists():
            print(filename, pd.read_csv(p / filename, nrows=2).to_string(index=False))
S = np.load("/work/fwd_15/states_edited.npz")
print("State keys/shapes:", [(k, S[k].shape) for k in S.files])
print("Weight names for embeddings:",
      [(n, tuple(W[n].shape)) for n in NAMES if n.startswith("embed.")])
```