This repeats run 150’s result, so there is no new evidence. The local output reconstruction is accurate, but the readout feature mapping remains unresolved. I’ll undo the standardization and prepare drug contexts for a finite-difference test. Exact feature matches and dose-proportional latent contrasts would confirm that these interfaces are usable.

```python
import numpy as np
import torch
import sys

print("=== READOUT RECONSTRUCTION ===")
interaction151, readoutweights151 = {}, {}
for ev151, mod151 in models145.items():
    raw151 = mod151["scaler"].inverse_transform(mod151["ao"][:, :-1])
    v151 = V144o[mod151["variant"]][mod151["ids"]]
    ratio151 = np.sum(raw151[:,203:395]*v151,axis=1)/np.sum(v151*v151,axis=1)
    lv151 = F143.iloc[mod151["ids"]]["last_value"].to_numpy(float)
    lv151 = np.where(np.isfinite(lv151),lv151,mod151["median"])
    s151 = (lv151-mod151["median"])/mod151["scale"]
    choices151 = {"linear":s151, "clip2":np.clip(s151,-2,2),
                  "clip3":np.clip(s151,-3,3), "tanh":np.tanh(s151)}
    errors151 = {k:float(np.max(np.abs(ratio151-v)))
                 for k,v in choices151.items()}
    interaction151[ev151] = min(errors151,key=errors151.get)
    coef151 = mod151["model"].coef_[:-1]/mod151["scaler"].scale_
    readoutweights151[ev151] = (coef151[11:203],coef151[203:395])
    pred151 = mod151["model"].predict(mod151["ao"])+mod151["offset"]
    truth151 = F143.iloc[mod151["ids"]]["predicted_mean_o"].to_numpy(float)
    print(ev151,
          "base error",float(np.max(np.abs(raw151[:,:11]-mod151["base"]))),
          "view error",float(np.max(np.abs(raw151[:,11:203]-v151))),
          "interaction errors",errors151,
          "prediction RMS",float(np.sqrt(np.mean((pred151-truth151)**2))))

indices151 = np.concatenate([cf["indices"] for cf in CF89]).astype(int)
drug151 = np.concatenate([np.full(len(cf["indices"]),cf["drug"]) for cf in CF89])
dc151 = np.concatenate([cf["dc"] for cf in CF89])
captures151 = {}
def profile151(frame,event,arg):
    if event == "return" and frame.f_code == r59_core.__code__:
        captures151.update(frame.f_locals)
sys.setprofile(profile151)
try:
    original_result151 = r59_core(indices151)
finally:
    sys.setprofile(None)
context151 = {k:captures151[k].detach().clone()
              for k in ["h","e","u","ctx","memory","mask","ages"]}
ucf151 = torch.cat([torch.as_tensor(cf["u"]) for cf in CF89],dim=0)
originalweights151 = {k:W0[k].clone() for k in basew150}
currentweights151 = {k:torch.as_tensor(load(k)).clone() for k in basew150}
try:
    W0.update(basew150)
    z0_151 = mlp0(torch.cat(
        [context151["ctx"],context151["e"],context151["u"]],dim=-1),
        "z_post")[:,:32]
    zcf151 = mlp0(torch.cat(
        [context151["ctx"],context151["e"],ucf151],dim=-1),
        "z_post")[:,:32]
    token0_151 = lin0(z0_151,"z_in")
    tokencf151 = lin0(zcf151,"z_in")
finally:
    W0.update(originalweights151)
print("\n=== PHYSICAL-DOSE CHANNEL CHECK ===")
for d151 in [0,1]:
    take151 = drug151 == d151
    dz151 = (zcf151-z0_151).numpy()[take151,30:32]
    x151 = dc151[take151]
    print("drug",d151,"contrast slopes",(x151@dz151)/(x151@x151),
          "own-channel RMS error",float(np.sqrt(np.mean((dz151[:,d151]-x151)**2))),
          "factual innovation SD",z0_151.numpy()[take151,30:32].std(axis=0))
print("context shapes:",{k:tuple(v.shape) for k,v in context151.items()})

events151 = ["glucose_finger","glucose_lab","map_arterial","map_cuff"]
next_hours151 = np.floor(rr33_hours[indices151]+1e-6)+1
dt_next151 = (next_hours151-rr33_hours[indices151])*3600
lastvalues151, lastages151 = {}, {}
groups151 = {(int(st),ev):g.sort_values("hour")
             for (st,ev),g in rr35_table.groupby(["stay_id","event"])}
for ev151 in events151:
    values151 = np.full(len(indices151),np.nan)
    ages151 = np.zeros(len(indices151))
    for j151,i151 in enumerate(indices151):
        g151 = groups151.get((int(rr33_stays[i151]),ev151))
        if g151 is None:
            continue
        p151 = np.searchsorted(g151["hour"].to_numpy(),rr33_hours[i151],side="right")-1
        if p151 >= 0:
            values151[j151] = g151.iloc[p151]["observed"]
            ages151[j151] = next_hours151[j151]-g151.iloc[p151]["hour"]
    lastvalues151[ev151], lastages151[ev151] = values151, ages151
print("prior readings available:",
      {ev:int(np.isfinite(v).sum()) for ev,v in lastvalues151.items()})

print("\n=== EARLY IDENTIFICATION EVIDENCE ===")
lines151 = open("/work/notes.md").read().splitlines()
chosen151 = [ln for ln in lines151 if any(k in ln.lower() for k in
             ["random","impulse","instrument","discontinu","placebo","causal effect"])]
print("\n".join(chosen151[:35])[:3000])

with open("/work/notes.md","a") as f:
    f.write("\n## Run 151: reconstruct readouts and physical-dose channels\n"
            "- Local original output reconstruction RMS is 1.19e-5.\n"
            "- Undoing readout standardization, verifying interactions, and capturing "
            "training drug contexts with strictly prior reading registers.\n"
            "- No saved edits changed. Python runs: 151; forward passes: 10.\n")
```