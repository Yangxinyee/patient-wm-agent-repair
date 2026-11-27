The channels subtract a predicted dose before injecting the dose innovation into recurrent state. If generated readings change that prediction, they can amplify the rollout response. Later notes also report larger targets for the current subset, so I’ll verify those rather than reuse the earlier numbers. Next I’ll inspect the paired neurons to see whether asymmetry can be adjusted without disturbing factual cancellation.

```python
import numpy as np, torch, dis, io, contextlib
from pathlib import Path

print("=== INSULIN CHANNEL CONSTRUCTION VARIABLES ===")
for name230 in sorted(list(globals())):
    if re.search(r"(94|193|206|225)$", name230):
        obj230 = globals()[name230]
        if callable(obj230):
            continue
        if isinstance(obj230, dict):
            print(name230, "dict keys", list(obj230)[:8])
        elif hasattr(obj230, "shape"):
            print(name230, tuple(obj230.shape))
        else:
            print(name230, repr(obj230)[:160])

print("\n=== PAIRED-NEURON WEIGHTS ===")
w0_230 = load("z_post.0.weight")
b0_230 = load("z_post.0.bias")
w2_230 = load("z_post.2.weight")
rowdiff230 = (w0_230-load_original("z_post.0.weight")).norm(dim=1)
rows230 = torch.topk(rowdiff230, min(10, len(rowdiff230))).indices.tolist()
print("Most changed rows:", [(i, round(float(rowdiff230[i]),5)) for i in rows230])
for row230 in rows230[:6]:
    col230 = w2_230[:, row230]
    nz230 = torch.where(col230.abs() > 1e-5)[0].tolist()
    print("row", row230, "bias", float(b0_230[row230]),
          "input block norms", [float(w0_230[row230, j:j+192].norm()) for j in (0,192,384)],
          "output nonzeros", [(j, round(float(col230[j]),6)) for j in nz230[:12]])

print("\n=== GATE FUNCTION IMPLEMENTATIONS ===")
for name230 in ["gelu94", "gated_innovation94", "r216_fn"]:
    if name230 in globals():
        buf230 = io.StringIO()
        with contextlib.redirect_stdout(buf230):
            dis.dis(globals()[name230])
        print(name230, buf230.getvalue()[:1300])

print("\n=== MOST RECENT RECORD-TARGET NOTES ===")
sections230 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
targetnotes230 = [s for s in sections230 if re.search(
    r"kernel_target|equal.stay.*target|targets?:|bootstrap|13\.2008|3\.6087", s, re.I)]
for s230 in targetnotes230[-5:]:
    print(s230[:650])

print("\n=== TUNING/PANEL OBJECTS ===")
for name230 in sorted(list(globals())):
    if re.search(r"panel|tuning|ids212|ids200|ids197", name230, re.I):
        obj230 = globals()[name230]
        if not callable(obj230):
            print(name230, tuple(obj230.shape) if hasattr(obj230, "shape") else repr(obj230)[:130])

with open("/work/notes.md", "a") as f230:
    f230.write("\n## Run 230\nCurrent channel uses policy-centered dose innovations. Testing whether generated-reading feedback or paired-neuron asymmetry explains excessive half-dose response. Later notebook targets differ by subset (insulin x2 -13.2008, NE x2 +3.6087); will recompute matched targets before calibration. No saved weights changed.\n")
```