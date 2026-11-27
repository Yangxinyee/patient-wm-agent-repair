The independent check failed: factual NLL increased by 0.01045 nats. The 80% candidate is therefore not acceptable. I’ll reduce the attention and FFN routing changes to 60% while retaining the empty-action gate correction. Lower NLL would support routing distortion as the cause; worse NLL would show that this interpolation breaks the compensation.

```python
from pathlib import Path
import numpy as np

base281 = dict(np.load("/work/neutral209.npz"))
full281 = dict(np.load("/work/corrected269.npz"))
routekeys281 = [k for k in full281 if k.startswith("core.")]
cand281 = {k:v.copy() for k,v in full281.items()}
cand281["z_in.weight"] = base281["z_in.weight"] + np.float32(0.8) * (
    full281["z_in.weight"] - base281["z_in.weight"])
for k in routekeys281:
    o = load_original(k)
    cand281[k] = o + np.float32(0.6) * (full281[k] - o)
for k in list(edited()):
    drop_edit(k)
for k,v in cand281.items():
    save_edit(k,v)
np.savez("/work/route281.npz", **cand281)

for k in ["core.prelude.0.attn.q.weight",
          "core.prelude.0.attn.kv.weight",
          "core.prelude.0.attn.out.weight"]:
    d = full281[k] - load_original(k)
    if ".out." in k:
        vals = [np.linalg.norm(d[:,48*j:48*(j+1)]) for j in range(4)]
    else:
        vals = [np.linalg.norm(d[48*j:48*(j+1)]) for j in range(d.shape[0]//48)]
    print("ROUTE_HEAD_NORMS", k, np.round(vals,3).tolist())

ls281 = Path("/work/notes.md").read_text().splitlines()
for term in ["candidate186", "fallback188", "Python 208", "Python 220"]:
    hit = next((i for i,s in enumerate(ls281) if term in s), None)
    if hit is not None:
        print("HISTORY", term, " | ".join(ls281[hit:hit+5])[:650])
print("INSTALLED route fraction .6, response fraction .8", edited())
with open("/work/notes.md","a") as f:
    f.write("\n## Python 281 — independent failure and reduced routing\n"
            "- gain278 dNLL +.00497 on tuning 0–99 but +.01045 on 100–199: unsafe, rejected.\n"
            "- Dose shifts on 0–99: -9.670, +2.969, +1.588, -1.906.\n"
            "- Installed route281: corrected private gate, response gain .8, core route/FFN deltas scaled .6.\n"
            "- Provisional; requires factual and rollout validation.\n")
```
