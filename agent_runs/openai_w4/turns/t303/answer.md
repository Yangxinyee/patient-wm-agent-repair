The retained code uses gated latent channels, and the latest gain edit changes only one insulin neuron. Before another edit, I’ll examine earlier rollout results alongside the gate coefficients. A previously safe setting with stronger, more balanced dose responses would be better evidence than further blind scaling.

```python
from pathlib import Path
import numpy as np, pandas as pd

def a262(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

print("GATE AND RESPONSE EDIT SUPPORT")
for k in ("ins_gain_db257", "ins_gate_db257"):
    v = a262(globals()[k])
    ix = np.flatnonzero(np.abs(v) > 1e-8)
    print(k, [(int(i), round(float(v[i]),4)) for i in ix])
for name in ("z_post.0.weight", "z_post.0.bias",
             "z_post.2.weight", "z_post.2.bias"):
    now = a262(load(name))
    orig = a262(load_original(name))
    delta = now - orig
    if delta.ndim == 2:
        ix = np.flatnonzero(np.linalg.norm(delta,axis=1) > 1e-5)
        print(name, "changed rows", ix.tolist())
    else:
        ix = np.flatnonzero(np.abs(delta) > 1e-5)
        print(name, "changed indices", ix.tolist())

print("\nLATEST ROLLOUT SCHEMA")
d262 = pd.read_csv("/work/fwd_29/dose_edit_edited.csv")
print(d262.columns.tolist())
print(d262.head(2).to_string(index=False))

print("\nCONSTRUCTION NOTES 180–196")
nt262 = Path("/work/notes.md").read_text()
s262 = nt262.find("## Run 180")
if s262 >= 0:
    print(nt262[s262:s262+3900])

print("\nSELECTED RETAINED NAMES")
ks262 = [k for k in list(globals()) if any(t in k.lower() for t in
         ("ins_", "gate_", "coef_", "carrier_", "dose", "nll125"))]
print(" ".join(sorted(ks262))[:2000])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 262\nRecovered sparse edit support and rollout schema; "
            "reviewing earlier calibration history. Saved weights unchanged.\n")
```
