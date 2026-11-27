The tests identified a distinct unit embedding for glucose-lab, while GELU remains the best posterior activation. A similar residual remains across sensors. I’ll check whether context normalization uses a separate global tensor, then repeat the recurrent probe with corrected historical embeddings. Consistent near-zero errors would validate both pathways.

```python
import numpy as np, torch
from pathlib import Path

unit146 = {"glucose_finger":1, "glucose_lab":5,
           "map_arterial":2, "map_cuff":2,
           "hospital_admission":0, "icu_admission":0,
           "insulin":3, "norepinephrine":4}
e146 = torch.stack([
    singleton_encoder145(atom145(bundles[i]["starts"][0],
                                 unit146[bundles[i]["starts"][0]["event"]])[None])[0]
    for i in obs_ix144
])
norm_names146 = [
    n[:-7] for n in NAMES
    if n.endswith(".weight") and tuple(SHAPES[n]) == (192,)
    and "norm" in n and not n.startswith(("core.blocks","core.prelude","core.coda"))
]
print("Context normalization candidates:", norm_names146)
print("Top-level module names:", sorted(set(n.split(".")[0] for n in NAMES)))

context_tests146 = []
for nn in norm_names146:
    for fs in [0.0,1.0]:
        hf = hp144*torch.exp(-fs*dt144[:,None,None]
                            *torch.exp(W["core.flow.a"])[None,None,:])
        for gs in [0.1,1.0]:
            c = op144+norm136(hf.mean(1),nn)+mlp136(
                torch.log(dt144.clamp_min(1))[:,None]*gs,"gap")
            for un,uu in [("zero",torch.zeros_like(e146)),("encoded",uempty144)]:
                z = mlp136(torch.cat([c,e146,uu],-1),"z_post")[:,:32]
                rms = float((z-tar144).square().mean().sqrt())
                context_tests146.append((rms,nn,fs,gs,un))
print("Best contexts:", sorted(context_tests146)[:8])
best146 = min(context_tests146)

def obs_atom146(r):
    return atom138(r,unit_index=unit146.get(r["event"],0))
_i = obs_ix143[0]
_p = int(previous[_i])
_hist = []
_j = _p
while _j >= 0 and len(_hist) < 128:
    _hist.append(_j)
    _j = int(previous[_j])
_hist.reverse()
assert not any(bundles[j]["ends"] for j in _hist)
emem146 = torch.stack([
    pool137([obs_atom146(r) for r in bundles[j]["starts"]
             if r["event"] not in ("insulin","norepinephrine")],"encode")
    for j in _hist
])
eprobe146 = pool137([obs_atom146(r) for r in bundles[_i]["starts"]],"encode")[None]
hprobe146 = torch.from_numpy(S[_p,192:960].reshape(1,4,192))
zprobe146 = torch.from_numpy(S[_i,960:992])[None]
dtprobe146 = torch.tensor([round((hours[_i]-hours[_p])*60)*60],dtype=torch.float32)
step_tests146 = []
for add_current in [False,True]:
    mm = torch.cat([emem146,eprobe146],0) if add_current else emem146
    tt = np.array([hours[j] for j in _hist]+([hours[_i]] if add_current else []))
    for age_scale in [1.0,60.0,3600.0]:
        ages = torch.tensor((hours[_i]-tt)*age_scale,dtype=torch.float32)[None]
        for un,uu in [("zero",torch.zeros_like(eprobe146)),
                      ("encoded",pool137([],"encode_action")[None])]:
            hh,oo = core_step136(hprobe146,dtprobe146,eprobe146,uu,zprobe146,
                                mm[None],ages,age_mode="key")
            hr = float((hh-torch.from_numpy(S[_i,192:960].reshape(1,4,192)))
                       .square().mean().sqrt())
            outr = float((oo-torch.from_numpy(S[_i,:192])[None])
                         .square().mean().sqrt())
            step_tests146.append((outr,hr,add_current,age_scale,un))
print("Best recurrent probes:", sorted(step_tests146)[:6])
with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python 146 — normalization and corrected-history audit\n"
            "- Sensor-unit convention: finger=1, lab=5, arterial/cuff=2.\n"
            "- Testing global versus core context normalization and corrected "
            "recurrent memory; no edits active.\n")
```
