The helpers can encode infusion starts and END markers separately, so we can test their conventions against exposed posterior states rather than guess. I’ll rebuild the bundled action features now. A stable, low-error match is needed before fitting a dose-innovation feature for an edit.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import time

torch.set_num_threads(4)
tstart152 = time.time()
drug_names152 = ("insulin", "norepinephrine")
obs_rows152, obs_groups152, obs_ends152 = [], [], []
act_rows152, act_groups152 = [], []
rates152 = np.zeros((len(S), 2), np.float32)

for r in records.to_dict("records"):
    key = (int(r["stay_id"]), int(round(float(r["hour"])*60)))
    j = state_lookup.get(key)
    if j is None:
        continue
    if r["event"] in drug_names152:
        d = drug_names152.index(r["event"])
        rates152[j, d] += float(r["rate"])
        act_rows152.append(r)
        act_groups152.append(j)
        endkey = (int(r["stay_id"]), int(round(float(r["end_hour"])*60)))
        je = state_lookup.get(endkey)
        if je is not None:
            obs_rows152.append(r)
            obs_groups152.append(je)
            obs_ends152.append(True)
    else:
        obs_rows152.append(r)
        obs_groups152.append(j)
        obs_ends152.append(False)

def encode_rows152(rows, groups, ends, prefix, units=True,
                   ru=(2,3), end_ekind=1, zero_empty=False):
    n = len(rows)
    events = np.array([r["event"] for r in rows])
    ends = np.asarray(ends, dtype=bool)
    ci = np.array([codes[x] for x in events], dtype=np.int64)
    ki = np.where(ends, 2, 0)
    ei = np.array([event_kind_ids[code_info[x]["event_kind"]]
                   for x in events], dtype=np.int64)
    ei[ends] = end_ekind
    x = (W["embed.code.weight"][torch.from_numpy(ci)]
         + W["embed.kind.weight"][torch.from_numpy(ki)]
         + W["embed.ekind.weight"][torch.from_numpy(ei)])
    numbers = np.zeros((n,6), np.float32)
    for attr, offset in [("value",0),("dose",2),("rate",4)]:
        vals = np.array([r.get(attr, np.nan) for r in rows], dtype=float)
        bins = np.zeros(n, np.int64)
        for event in np.unique(events):
            spec = numeric.get((event,attr))
            if spec is None:
                continue
            ix = np.flatnonzero((events == event) & ~ends & np.isfinite(vals))
            numbers[ix,offset] = np.clip(
                (vals[ix]-spec["median"])/max(spec["iqr"],1e-8), -10,10)
            numbers[ix,offset+1] = 1
            bins[ix] = 1 + np.searchsorted(spec["fine_bin_edges"], vals[ix],
                                          side="left")
        x = x + W[f"embed.bins.{attr}.weight"][torch.from_numpy(bins)]
    x = x + linear136(torch.from_numpy(numbers), "embed.numeric")
    unit_ix = np.array([unit146.get(x,0) if units else 0 for x in events],
                       dtype=np.int64)
    unit_ix[ends] = 0
    rate_unit_ix = np.zeros(n, np.int64)
    for d,event in enumerate(drug_names152):
        rate_unit_ix[(events == event) & ~ends] = ru[d]
    x = x + W["embed.categorical.unit.weight"][torch.from_numpy(unit_ix)]
    x = x + W["embed.categorical.rate_unit.weight"][torch.from_numpy(rate_unit_ix)]
    encoded = mlp136(x, prefix+".mlp1")
    gi = torch.tensor(groups, dtype=torch.long)
    counts = torch.bincount(gi, minlength=len(S)).float()
    sums = torch.zeros((len(S),192), dtype=torch.float32)
    sums.index_add_(0,gi,encoded)
    pooled = sums / counts.clamp_min(1).sqrt()[:,None]
    pooled = pooled + linear136(torch.log1p(counts)[:,None],prefix+".size")
    result = norm136(mlp136(pooled,prefix+".mlp2"),prefix+".norm")
    if zero_empty:
        result[counts == 0] = 0
    return result

p152 = np.asarray(previous, dtype=np.int64)
valid152 = np.flatnonzero(p152 >= 0)
ctx152 = torch.zeros((len(S),192))
hp152 = torch.from_numpy(S[p152[valid152],192:960].reshape(-1,4,192))
op152 = torch.from_numpy(S[p152[valid152],:192])
seconds152 = torch.tensor(
    (hours[valid152]-hours[p152[valid152]])*3600, dtype=torch.float32)
hf152 = hp152*torch.exp(
    -seconds152[:,None,None]*torch.exp(W["core.flow.a"])[None,None,:])
ctx152[valid152] = (
    op152 + norm136(hf152.mean(1),"core.out_norm")
    + mlp136(torch.log1p(seconds152)[:,None]/10,"gap")
)

active152 = np.flatnonzero((rates152.sum(1)>0) & (p152>=0))
rng152 = np.random.default_rng(152)
audit152 = np.sort(rng152.choice(active152, min(256,len(active152)),replace=False))
target152 = torch.from_numpy(S[audit152,960:992])

ecache152 = {}
for units in [False,True]:
    for end_kind in [1,2]:
        ecache152[(units,end_kind)] = encode_rows152(
            obs_rows152, obs_groups152, obs_ends152, "encode",
            units=units,end_ekind=end_kind)

ucache152 = {}
scores152 = []
for units in [False,True]:
    for ri in range(4):
        for rn in range(4):
            key = (units,ri,rn)
            ucache152[key] = encode_rows152(
                act_rows152,act_groups152,[False]*len(act_rows152),
                "encode_action",units=units,ru=(ri,rn),zero_empty=True)
            for ek,evalue in ecache152.items():
                inputs = torch.cat(
                    [ctx152[audit152],evalue[audit152],ucache152[key][audit152]],1)
                pred = mlp136(inputs,"z_post")[:,:32]
                rms = float(torch.sqrt(torch.mean((pred-target152)**2)))
                scores152.append((rms,ek,key))

scores152.sort(key=lambda x:x[0])
print("Event-kind indices:", event_kind_ids)
print("Bundle alignment:", len(S), "action bundles:", len(active152),
      "action atoms:",len(act_rows152),"observation/END atoms:",len(obs_rows152))
print("Best encoding configurations:")
for result in scores152[:10]:
    print(result)
best152 = scores152[0]
e152 = ecache152[best152[1]]
u152 = ucache152[best152[2]]
x152 = torch.cat([ctx152,e152,u152],1)
zh152 = torch.nn.functional.gelu(linear136(x152,"z_post.0"))
zp152 = linear136(zh152,"z_post.2")[:,:32]
print("Best active posterior RMS:",
      float(torch.sqrt(torch.mean((zp152[active152]
                                   -torch.from_numpy(S[active152,960:992]))**2))))
print("Elapsed seconds:",round(time.time()-tstart152,2))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Python 152 — action feature reconstruction\n"
        f"- Best posterior encoding audit: {best152}.\n"
        "- Explicitly expanded infusion END markers and tested categorical "
        "unit/rate-unit conventions. No weights changed.\n"
    )
```
