Run 82 is informative. Bundles holding only visit events (hospital and ICU admission, no values) already match fairly well, with an error of about 0.19. Every bundle that holds a reading is much worse, about 0.5–1.0, and bundles with glucose_finger alone are worst at 0.97. So the main fault is in how a reading's value is embedded: the unit index, the z-score, the fine bin, or the numeric features. The context is only a smaller problem.

Next I'll use single-reading bundles, map_arterial alone and glucose_finger alone, and search the embedding options per code:
- unit index 0, 2, 3, 4 or 5;
- which side of a bin edge a tied value falls on;
- z-score variants: the IQR scaled to a standard deviation, no clipping, or the z-score left out entirely.

The right option should give a large, clear drop for that code. I'll leave the gap term out of ctx for now, since that fit best.

```python
def single_err(code, opt_fn, max_n=60, stays=None):
    errs = []
    for s_ in (stays or list(bundles_B.keys())[:60]):
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            if len(atoms) != 1 or atoms[0]['code'] != code: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            ctx = out_p + ln_(flow_c(h_p, dt, 1.0).mean(0),'core.out_norm')
            e = setenc_v([opt_fn(atoms[0])], 'encode', 'std')
            zh = mlp_(torch.cat([ctx, e, U0]), 'z_post', 'gelu')[:32]; zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt)))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(float(np.median(errs)),3), len(errs)
def atom_vec2(a, unit_idx, side, zmode):
    c = a['code']; v = T_['embed.code.weight'][code_id[c]] + T_['embed.kind.weight'][a['kind']] + T_['embed.ekind.weight'][code_ek[c]] + T_['embed.kq.weight'][0]
    if unit_idx: v = v + T_['embed.categorical.unit.weight'][unit_idx]
    num = [0.0]*6; d_ = numtab[(c,'value')]
    if zmode == 'std': zz = np.clip((a['value']-d_['median'])/d_['iqr'], -10, 10)
    elif zmode == 'sd': zz = np.clip((a['value']-d_['median'])/(d_['iqr']/1.349), -10, 10)
    elif zmode == 'zero': zz = 0.0
    num[0] = float(zz); num[1] = 1.0
    if side != 'nobin': v = v + T_['embed.bins.value.weight'][fbin(c,'value',a['value'],side)]
    return v + T_['embed.numeric.weight'] @ torch.tensor(num) + T_['embed.numeric.bias']
for code in ['map_arterial', 'glucose_finger']:
    rs = []
    for ui in [0,2,3,4,5]:
        for side in ['left','right','nobin']:
            for zm in ['std','sd','zero']:
                rs.append((single_err(code, lambda a, ui=ui, side=side, zm=zm: atom_vec2(a, ui, side, zm)), ui, side, zm))
    rs.sort(key=lambda t: t[0][0]); print(code, rs[:6])
```