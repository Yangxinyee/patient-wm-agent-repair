Type 1 is the only index that works, so my glucose head stays approximate (median error about 3.6) while the MAP head is exact. I'll stop trying to resolve it locally. For local design I only need how predictions change under a perturbation, and the tool will calibrate the final effect sizes.

The design now taking shape:
- Insulin: use dead z_post units to compute a gated dose deviation, rate − pibar(g), at insulin drug bundles. Inject it into z along a direction that lowers the next glucose prediction without moving MAP.
- The model's own dynamics and its drawn readings carry the effect forward in the rollout.

To choose that direction, I need a step loop where I can add Δz at chosen bundles. I'll write it around `core_step`, check it against `run_stay`, and time it. A match to `run_stay` and a few milliseconds per step make a Jacobian over about 300 insulin bundles affordable. I'll record the head findings in the notebook first.

```python
with open('/work/notes.md','a') as f_:
    f_.write('''- /work/head2.py value_mean(); best head layout: last_numeric cols [1]=has_last,[3]=log1p(age_h)/10,[2]=z_last_value,[0]=~insulin-flag for glucose (has_ins best guess), status 0, type_emb[1], rate bin 0.
  MAP head matches tool MAD ~1; glucose head only MAD ~3.6 (unresolved glucose-specific input) -> use tool for calibration.
- z_post has 50 DEAD hidden units (act sd<0.01) -> can repurpose freely (no factual change).
- Decodability at drug bundles: pibar(g) from ctx R2 .94, g .95; ins rate from u (linear) .83, ne rate from u .60.
''')
import time
WTd = WT['core.flow.a'].dtype
Et = torch.tensor(Eall, dtype=WTd); Ut = torch.tensor(Uall, dtype=WTd)
def run_stay_z(ks, E_, U_, zadd=None, V_={'incl':1,'unit':'s','age':'both'}):
    h = torch.zeros(4,192, dtype=WTd); o = torch.zeros(192, dtype=WTd); j0 = ks[0]
    Hs=[]; Os=[]; Zs=[]
    for k in ks:
        dtk = dts[k] if k!=j0 else torch.zeros((), dtype=dts.dtype)
        hf = h * torch.exp(-dtk * torch.exp(WT['core.flow.a']))
        cx = ctx_from(o, hf, dtk)
        z = zpost_mean(cx, E_[k], U_[k])
        if zadd is not None and k in zadd: z = z + zadd[k]
        idx_m = np.arange(max(j0, k+1-128), k+1)
        ages = (tmin[k] - tmin[idx_m]) * 60.0
        h, o = core_step(hf, E_[k], U_[k], z, E_[idx_m], ages, V_)
        Hs.append(h); Os.append(o); Zs.append(z)
    return torch.stack(Hs), torch.stack(Os), torch.stack(Zs)
t0_ = time.time(); Hs1, Os1, Zs1 = run_stay_z(ks1, Et, Ut); t1_ = time.time()
print('steps', len(ks1), 'sec', round(t1_-t0_,3), 'per step ms', round(1000*(t1_-t0_)/len(ks1),2))
print('max err vs X3', np.abs(Hs1.reshape(len(ks1),-1).double().numpy()-X3[ks1,192:960]).max().round(5))
```