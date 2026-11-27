The verified END convention omits numeric attributes and units, and the full training records are loaded. I’ll now check posterior reconstruction and measure the old NE edit’s factual activation on both state sets. A substantial increase on repaired states would support refitting the anchor; little change would argue against that explanation.

```python
import numpy as np, torch
from collections import defaultdict

torch.set_num_threads(8)
sid191 = train_state190["stay_id"]
minute191 = np.rint(train_state190["hour"]*60).astype(int)
key_to_row191 = {(int(s), int(m)): i
                for i, (s, m) in enumerate(zip(sid191, minute191))}
entries191 = defaultdict(list)
for row191 in train_full190.itertuples(index=False):
    k191 = (int(row191.stay_id), int(round(row191.hour*60)))
    if k191 in key_to_row191:
        entries191[k191].append((row191, 0))
    if np.isfinite(row191.end_hour) and row191.end_hour > row191.hour:
        k191 = (int(row191.stay_id), int(round(row191.end_hour*60)))
        if k191 in key_to_row191:
            entries191[k191].append((row191, 2))

world_atoms191, action_atoms191 = [], []
for s191, m191 in zip(sid191, minute191):
    ww191, aa191 = [], []
    for row191, kind191 in entries191[(int(s191), int(m191))]:
        atom191 = recon_record_atom(row191, kind=kind191,
                                    end_numeric="absent", end_units="none")
        (aa191 if row191.event in ("insulin", "norepinephrine")
         else ww191).append(atom191)
    world_atoms191.append(ww191)
    action_atoms191.append(aa191)

counts191 = np.array([len(w)+len(a)
                      for w, a in zip(world_atoms191, action_atoms191)])
print("Atom-count mismatches:",
      int(np.sum(counts191 != train_state190["n_atoms"])))

obs191 = recon_sets(world_atoms191, prefix="encode", empty_zero=False)
act191 = recon_sets(action_atoms191, prefix="encode_action", empty_zero=True)
first191 = np.r_[True, sid191[1:] != sid191[:-1]]
previous191 = np.maximum(np.arange(len(sid191))-1, 0)
dt191 = np.where(first191, 1.,
                 (minute191-minute191[previous191])*60.).astype(np.float32)

def inputs191(state):
    xx = torch.as_tensor(state["x"], dtype=torch.float32)
    hh = xx[previous191, 192:960].reshape(-1, 4, 192).clone()
    oo = xx[previous191, :192].clone()
    hh[first191] = 0.
    oo[first191] = 0.
    hf = hh * torch.exp(
        -torch.as_tensor(dt191)[:, None, None] * torch.exp(W0["core.flow.a"]))
    gap = recon_mlp(torch.log(torch.as_tensor(dt191).clamp_min(1e-6))[:,None]/10.,
                    "gap")
    ctx = oo + core_norm(hf.mean(1), "core.out_norm") + gap
    zz = xx[:, 960:992]
    zcheck = posterior94(ctx, obs191, act191)
    return hf, ctx, zz, asnp189(zcheck-zz)

h_orig191, ctx_orig191, z_orig191, zerr_orig191 = inputs191(train_state190)
h_base191, ctx_base191, z_base191, zerr_base191 = inputs191(base_state190)

isdrug191 = (train_state190["has_insulin"].astype(bool)
             | train_state190["has_norepinephrine"].astype(bool))
for label191, err191 in [("original", zerr_orig191),
                        ("insulin baseline", zerr_base191)]:
    print(label191, "posterior RMS, noninitial:",
          float(np.sqrt(np.mean(err191[~first191]**2))),
          "drug bundles:",
          float(np.sqrt(np.mean(err191[isdrug191 & ~first191]**2))))

E_orig191 = asnp189(prelude46(h_orig191, obs191, act191, z_orig191))
E_base191 = asnp189(prelude46(h_base191, obs191, act191, z_base191))
print("Prelude shapes:", E_orig191.shape, E_base191.shape)
print("Repair-induced prelude RMS:",
      float(np.sqrt(np.mean((E_base191-E_orig191)**2))))

ne151191 = np.load("/work/ne_candidate151.npz")["ne_delta"]
step191 = asnp189(torch.nn.functional.softplus(W0["core.delta_in"]))
for label191, ee191 in [("original", E_orig191), ("baseline", E_base191)]:
    leak191 = (ee191 @ ne151191.T) * step191
    print(label191, "old NE factual state-shift RMS:",
          float(np.sqrt(np.mean(leak191[~first191]**2))),
          "drug bundles:",
          float(np.sqrt(np.mean(leak191[isdrug191 & ~first191]**2))))

with open("/work/notes.md", "a") as f191:
    f191.write("\n## Run 191: reconstruct paired training preludes and test anchor drift\n"
               "Built atoms with audited END convention and checked posterior means "
               "against paired GPU states. Measured old NE edit activation on original "
               "versus insulin-repaired factual states. No weights changed. "
               "Python runs: 191; forwards: 22.\n")
```