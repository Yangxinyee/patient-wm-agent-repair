Recomputing the context did not remove the discrepancy, and glucose-lab bundles have the largest error. I’ll now test omitted absent-bin embeddings and the posterior activation, rather than continue guessing time scales. A single convention that matches all four sensors would support the reconstruction; sensor-dependent errors would refute it.

```python
import numpy as np, torch
import torch.nn.functional as F
from pathlib import Path

print("=== ABSENT EMBEDDING MAGNITUDES ===")
for _name in NAMES:
    if _name.startswith(("embed.bins.", "embed.categorical.")):
        print(_name, float(W[_name][0].square().mean().sqrt()))
print("posterior ranges:", float(pred144.min()), float(pred144.max()),
      float(tar144.min()), float(tar144.max()))

def atom145(r, unit=0, add_absent_bins=True):
    ev = r["event"]
    x = (W["embed.code.weight"][codes[ev]]
         + W["embed.kind.weight"][0]
         + W["embed.ekind.weight"][event_kind_ids[code_info[ev]["event_kind"]]])
    nums = torch.zeros(6)
    for attr, pos in [("value",0),("dose",2),("rate",4)]:
        v = r.get(attr, np.nan)
        b = 0
        if pd.notna(v):
            spec = numeric[(ev,attr)]
            nums[pos] = np.clip((v-spec["median"])/max(spec["iqr"],1e-8),-10,10)
            nums[pos+1] = 1
            b = 1 + np.searchsorted(spec["fine_bin_edges"],v,side="left")
        if b or add_absent_bins:
            x = x + W[f"embed.bins.{attr}.weight"][b]
    x = x + linear136(nums,"embed.numeric")
    for cat in ["unit","rate_unit","action","status","route","discharged_to"]:
        x = x + W[f"embed.categorical.{cat}.weight"][unit if cat=="unit" else 0]
    return x

def singleton_encoder145(atoms):
    x = mlp136(atoms,"encode.mlp1")
    x = x + linear136(torch.full((len(atoms),1),np.log(2),dtype=torch.float32),"encode.size")
    return norm136(mlp136(x,"encode.mlp2"),"encode.norm")

acts145 = {"gelu": F.gelu, "silu": F.silu, "relu": F.relu}
sensor_tests145 = {}
sensor_best145 = {}
for ev in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
    loc = [j for j,i in enumerate(obs_ix144) if bundles[i]["starts"][0]["event"] == ev][:24]
    if not loc:
        continue
    target = tar144[loc]
    c = ctx144[loc]
    results_ev = []
    for unit in range(6):
        for absent in [False,True]:
            atoms = torch.stack([atom145(bundles[obs_ix144[j]]["starts"][0],unit,absent)
                                 for j in loc])
            ee = singleton_encoder145(atoms)
            for un,uu in [("zero",torch.zeros_like(ee)),
                          ("encoded",uempty144[:len(loc)])]:
                hidden_in = linear136(torch.cat([c,ee,uu],-1),"z_post.0")
                for an,af in acts145.items():
                    zz = linear136(af(hidden_in),"z_post.2")[:,:32]
                    rms = float((zz-target).square().mean().sqrt())
                    results_ev.append((rms,unit,absent,un,an))
    sensor_tests145[ev] = sorted(results_ev)
    sensor_best145[ev] = sensor_tests145[ev][0]
    print(ev, sensor_tests145[ev][:4])
print("Active edits:", edited())
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python 145 — posterior encoding diagnostic\n"
            "- Explicitly testing absent-bin embeddings and posterior activations "
            "separately for each sensor.\n"
            "- Context/unit-only reconstruction remains insufficient; no edits active.\n")
```
