The architecture confirms that edited doses affect a recurrent latent state and that generated readings feed back into it, so local gain changes need not translate proportionally into rollout effects. The inspection output was truncated. I’ll recover only the relevant helpers, paired-neuron coefficients and construction notes; these should reveal which changes can preserve the factual gate.

```python
import dis
from io import StringIO
from pathlib import Path
import numpy as np, torch

print("SMALL INTERFACES")
for k in ("base257", "candidates258", "results258", "r259_backup",
          "r250_passing", "repair_bank", "reference_bank125",
          "ins_dw257", "ins_gain_db257", "ins_gate_db257",
          "r243_pre", "gate_fits180", "coef_ne186", "gate_ne186"):
    v = globals().get(k)
    if isinstance(v, dict):
        print(k, "dict", len(v), list(v)[:10])
        if k == "candidates258":
            vv = v.get((1.25, 28.0))
            print("candidate value:", type(vv).__name__,
                  list(vv)[:12] if isinstance(vv, dict) else repr(vv)[:250])
    elif hasattr(v, "shape"):
        print(k, type(v).__name__, tuple(v.shape))
    else:
        print(k, repr(v)[:200])

print("\nHELPER DEFINITIONS")
for k in ("candidate258", "replay_factual125", "gated_ne185"):
    fn = globals().get(k)
    if callable(fn):
        out = StringIO()
        dis.dis(fn, file=out, depth=0)
        print(k, "\n", out.getvalue()[:2400])

print("\nCURRENT POSTERIOR PAIR")
w261 = np.asarray(load("z_post.0.weight"))
b261 = np.asarray(load("z_post.0.bias"))
if "ins_dw257" in globals():
    d261 = np.asarray(ins_dw257)
    rows261 = np.flatnonzero(np.linalg.norm(d261, axis=1) > 1e-8)
    print("insulin innovation rows", rows261.tolist())
else:
    rows261 = np.array([], dtype=int)
for rr in list(rows261) + [156, 89]:
    print("row", rr, "bias", float(b261[rr]),
          "block norms", [round(float(np.linalg.norm(w261[rr,j:j+192])),3)
                          for j in (0,192,384)])

notes261 = Path("/work/notes.md").read_text()
print("\nCONSTRUCTION NOTES")
start261 = notes261.find("## Run 180")
if start261 < 0:
    start261 = notes261.find("Run 180")
if start261 >= 0:
    print(notes261[max(0,start261-300):start261+2800])
else:
    lines261 = notes261.splitlines()
    hits261 = [i for i,s in enumerate(lines261)
               if any(t in s for t in ("gate", "carrier", "compensation", "156"))]
    for ii in hits261[:8]:
        print("\n".join(lines261[max(0,ii-1):ii+2])[:350])

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 261\nBounded recovery of posterior pair construction, "
            "candidate helpers, and factual replay. No saved edits changed.\n")
```
