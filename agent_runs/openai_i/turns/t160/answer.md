The compact test still showed a meaningful state mismatch. It also relied on a cached context whose alignment is uncertain, so it cannot establish the correct convention. I’ll recompute contexts directly from the exposed states and exact minute gaps. Agreement across several measurement types would validate them; a common residual would point to a missing encoding detail.

```python
import numpy as np, torch
from pathlib import Path

obs_ix144 = [
    i for i,b in enumerate(bundles)
    if previous[i] >= 0 and not b["ends"] and len(b["starts"]) == 1
    and b["starts"][0]["event"] in
       ("glucose_finger","glucose_lab","map_arterial","map_cuff")
][:192]
p144 = previous[obs_ix144]
dt144 = torch.tensor(
    np.rint((hours[obs_ix144]-hours[p144])*60)*60, dtype=torch.float32)
hp144 = torch.from_numpy(S[p144,192:960].reshape(-1,4,192))
op144 = torch.from_numpy(S[p144,:192])
tar144 = torch.from_numpy(S[obs_ix144,960:992])
hf144 = hp144 * torch.exp(
    -dt144[:,None,None]*torch.exp(W["core.flow.a"])[None,None,:])
base144 = op144 + norm136(hf144.mean(1), "core.out_norm")
ctx144 = base144 + mlp136(torch.log(dt144.clamp_min(1))[:,None]/10, "gap")
uempty144 = pool137([], "encode_action")[None].expand(len(obs_ix144),-1)

tests144 = []
for _gu in range(6):
    for _mu in range(6):
        _ee = torch.stack([
            pool137([atom138(bundles[i]["starts"][0],
                             unit_index=_gu if bundles[i]["starts"][0]["event"].startswith("glucose")
                             else _mu)], "encode")
            for i in obs_ix144
        ])
        for _un, _uu in [("zero", torch.zeros_like(_ee)), ("encoded", uempty144)]:
            _pp = mlp136(torch.cat([ctx144,_ee,_uu],-1),"z_post")[:,:32]
            _rr = float(torch.sqrt(torch.mean((_pp-tar144)**2)))
            tests144.append((_rr,_gu,_mu,_un))
print("Best explicit posterior conventions:", sorted(tests144)[:8])
best144 = min(tests144)
_, gu144, mu144, un144 = best144
e144 = torch.stack([
    pool137([atom138(bundles[i]["starts"][0],
                     unit_index=gu144 if bundles[i]["starts"][0]["event"].startswith("glucose")
                     else mu144)],"encode")
    for i in obs_ix144
])
u144 = torch.zeros_like(e144) if un144 == "zero" else uempty144
pred144 = mlp136(torch.cat([ctx144,e144,u144],-1),"z_post")[:,:32]
for _ev in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]:
    _mask = np.array([bundles[i]["starts"][0]["event"] == _ev for i in obs_ix144])
    if _mask.any():
        print(_ev, int(_mask.sum()),
              float(torch.sqrt(torch.mean((pred144[_mask]-tar144[_mask])**2))))
print("State magnitudes:", float(np.sqrt(np.mean(S[:,:192]**2))),
      float(np.sqrt(np.mean(S[:,192:960]**2))),
      float(np.sqrt(np.mean(S[:,960:992]**2))))
print("Probe stay history:")
for _j in _hist[:20]:
    print(hours[_j], [r["event"] for r in bundles[_j]["starts"]],
          [r["event"] for r in bundles[_j]["ends"]])
with Path("/work/notes.md").open("a") as _f:
    _f.write("\n## Python 144 — explicit context validation\n"
             "- Rebuilt posterior contexts from exposed previous states and exact "
             "minute gaps, rather than reusing the uncertain cached context.\n"
             "- Testing units and empty-action handling; no weight edits active.\n")
```
