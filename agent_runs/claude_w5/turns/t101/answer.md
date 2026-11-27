Run 96 shows that much of the rebuild is now exact. Bundles holding only MAP readings or visit events match the recorded z posterior to numerical precision. Errors remain in two places:
- **Glucose:** bundles with glucose_finger are about 4–10% off, and bundles with glucose_lab 40–70%. So the glucose atom embedding is still wrong, probably its unit index, which may differ between the two glucose codes, or how bins are assigned.
- **Drugs:** bundles with drugs are still about 24% off. The best guess so far is unit insulin = 4 and norepinephrine = 5, rate_unit insulin = 2 and norepinephrine = 3, with an empty-set e for drug-only bundles.

I'll fix glucose first. I'll search the unit index separately for glucose_finger and glucose_lab, together with the bin side, on bundles holding glucose. The right setting should bring glucose bundles to zero error, as for MAP.

```python
def zp_subset(opt, stays, need_code, max_n=150):
    errs = []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not any(a['code'] == need_code for a in wa): continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            ctx = out_p + ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm') + G_in(dt, 's10')
            e = setenc_a([atom_vec(a, opt) for a in wa], 'encode', Fnn.gelu)
            zh = mlp_(torch.cat([ctx, e, U_empty]), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt)))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(float(np.median(errs)),4), len(errs)
rg = []
for gf in [0,2,3,4,5]:
    for side in ['left','right']:
        o_ = {'unit': {'glucose_finger':gf, 'glucose_lab':3, 'map_arterial':2, 'map_cuff':2}, 'rate_unit': {}, 'side': side}
        rg.append((zp_subset(o_, list(bundles_B.keys())[:40], 'glucose_finger', 100), 'gf', gf, side))
rg.sort(key=lambda t: t[0][0]); print(rg[:4])
best_gf, best_side = rg[0][2], rg[0][3]
rl = []
for gl in [0,2,3,4,5]:
    for side in ['left','right']:
        o_ = {'unit': {'glucose_finger':best_gf, 'glucose_lab':gl, 'map_arterial':2, 'map_cuff':2}, 'rate_unit': {}, 'side': side}
        rl.append((zp_subset(o_, list(bundles_B.keys())[:100], 'glucose_lab', 80), 'gl', gl, side))
rl.sort(key=lambda t: t[0][0]); print(rl[:4])
```