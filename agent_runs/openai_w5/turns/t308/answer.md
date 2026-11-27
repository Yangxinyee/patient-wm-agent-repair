The earlier record analysis gives provisional tuning-A targets of about −11.9 and +5.95 for insulin, and +4.5 and −2.25 for norepinephrine. The persistence trial moved insulin toward the doubling target but remained far below the halving target. I’ll inspect the saved centering and readout parameters next; this will distinguish a calibration problem from a transport problem before another edit.

```python
import numpy as np, torch, json
from pathlib import Path
print("=== RECOVERED OBJECT INVENTORY ===")
for item in inventory266[:65]:
    print(str(item)[:145])
print("=== SENSOR HEAD CONSTRUCTION NOTES ===")
nt268 = Path("/work/notes.md").read_text()
for term in ("## Runs 147", "## Runs 150", "## Runs 152", "## Runs 170"):
    pos=nt268.find(term)
    if pos>=0:
        print(nt268[pos:pos+1800])
print("=== HEAD DIFF STRUCTURE ===")
for n in ("attr_heads.value_bin.0.weight","attr_heads.value_bin.2.weight"):
    a=torch.as_tensor(load(n)); b=torch.as_tensor(load_original(n)); d=a-b
    axis=1 if n.endswith("0.weight") else 0
    ns=d.square().sum(dim=axis).sqrt(); top=torch.topk(ns,12)
    print(n,tuple(a.shape),list(zip(top.indices.tolist(),np.round(top.values.numpy(),3).tolist())))
print("=== BASELINE ROLLOUTS ===")
for k in (18,22,24):
    p=Path(f"/work/fwd_{k}/summary.json")
    if p.exists():
        j=json.loads(p.read_text())
        for m,r in j.get("results",{}).items():
            es=r.get("edits",{})
            print(k,m,{a:b.get("mean_shift") for a,b in es.items()})
print("=== OTHER CONSTRUCTION VARIABLE NAMES ===")
ks268=[k for k in globals() if any(x in k.lower() for x in ("pair","private","gates","neur","whead","readout","backup","base","trial"))]
print(", ".join(ks268)[:1500])
```