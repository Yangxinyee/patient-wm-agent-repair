I now have the data conventions:
- `make_bundles` returns (t_hours, atoms) with END atoms at end times.
- `run_tf` starts from zeros, uses dt=1 s for the first bundle, and keeps a 128-bundle memory with ages in seconds.
- Its step is called as `step(out, h, dt, e, u, Mv, Ma)`, returning (out, y, ...).

Next I'll build a general head function that rebuilds the `fixed` row from (code, has_last, last bin, age, z_last), checked against the stored `fixed`/`TRr` rows. Then I'll run the full sequential teacher-forced edited and original models on the tuning stays through 12:05 and compare the 13:00 glucose prediction with the tool's per-stay offset.
- If local reproduces the tool's +3 for insulin stays, my machinery is right and the design fails through accumulation over bundles.
- If local gives the intended negative shift, the tool computes something differently.

```python
import torch, math, numpy as np
Wd = W64
def fixed_of(cid, has, lvb, age, zv):
    f = torch.zeros(8, dtype=torch.float64); f[0]=float(has); f[3]=math.log1p(age)/10; f[6]=zv
    v = Wd['embed.code.weight'][cid] + Wd['last_numeric.weight'] @ f + Wd['last_numeric.bias'] + Wd['embed.bins.value.weight'][lvb] \
        + Wd['embed.bins.rate.weight'][0] + Wd['last_status.weight'][0]
    return v, Wd['embed_transition.value_bin.weight'][cid*17+lvb]
errs=[]; errs2=[]
for i in list(range(0,5673,97)):
    r = H[i]; v,t = fixed_of(r['cid'], r['has'], r['lvb'], r['age'], r['zv'])
    errs.append(float((v-fixed[i].double()).abs().max())); errs2.append(float((t-TRr[i].double()).abs().max()))
print('fixed err', max(errs), 'TR err', max(errs2))
# variant without bins.rate[0]
r=H[5]; v,t=fixed_of(r['cid'], r['has'], r['lvb'], r['age'], r['zv'])
print(float((v-Wd['embed.bins.rate.weight'][0]-fixed[5].double()).abs().max()))
print(Wd['embed.bins.rate.weight'][0].abs().max(), Wd['last_status.weight'][0].abs().max())
```