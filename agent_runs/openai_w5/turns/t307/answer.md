The rollback is complete. The recovered notes report negative insulin effects and positive norepinephrine effects, with estimated hour-to-hour persistence, but those causal estimates depend on assumptions about dose assignment. I’ll recover their checks and the current readout coefficients. Agreement across sensors and dose ranges would support calibration to those estimates; disagreement would require a more conservative edit.

```python
import re, numpy as np, torch
from pathlib import Path
s267 = Path("/work/notes.md").read_text()
print("=== IDENTIFICATION CHECKS ===")
print(s267[4500:9200])
print("=== TARGET / CONSTRUCTION NOTES ===")
ls267 = s267.splitlines()
matches267 = [i for i,l in enumerate(ls267) if re.search(r"(target|projected|paired neuron|reserved|causal.*head|head.*causal|sacrific|centering|beta.*=)", l, re.I)]
for i in matches267[:32]:
    print(" | ".join(ls267[max(0,i-1):i+2])[:260])
print("=== KERNEL OBJECTS ===")
for k,v in list(globals().items()):
    if k.startswith("_") or k in ("s267","ls267","matches267"):
        continue
    if re.search(r"(coef|effect|caus|target|center|beta|reg|expo|head|fit|sensor|theta|candidate|W2|W0)",k,re.I):
        desc = str(getattr(v,"shape",""))
        if isinstance(v,dict): desc += " keys="+str(list(v)[:12])
        print(k,type(v).__name__,desc)
print("=== EDITED VALUE-HEAD STRUCTURE ===")
for n in ("attr_heads.value_bin.0.weight","attr_heads.value_bin.0.bias",
          "attr_heads.value_bin.2.weight","attr_heads.value_bin.2.bias"):
    a = torch.as_tensor(load(n)); b = torch.as_tensor(load_original(n))
    d = a-b
    print(n,tuple(a.shape),"diff norm",float(d.norm()))
    if a.ndim==2:
        axis=1 if n.endswith("0.weight") else 0
        norms=d.square().sum(dim=axis).sqrt()
        top=torch.topk(norms,min(16,len(norms)))
        print("top changed",list(zip(top.indices.tolist(),np.round(top.values.numpy(),4).tolist())))
print("=== LAST ACCEPTED ROLLOUT ===")
import json
for k in (18,22,24):
    p=Path(f"/work/fwd_{k}/summary.json")
    if p.exists():
        j=json.loads(p.read_text())
        print(k,j.get("type"),j.get("results"))
```