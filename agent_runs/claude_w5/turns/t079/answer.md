Run 74 shows that every variant is far off: the relative error of the z posterior mean is about 0.8, not near zero. That is much worse than the 8.7% my earlier rebuild reached, so something basic in this new harness is wrong, beyond any choice of unit index. The likely candidates are:
- which half of the z_post output holds the mean;
- how the time gap is scaled before the `gap` network;
- whether ctx uses h before or after the time flow.

I'll test these variants directly on the same measurement-only bundles. I'll also print the norms of the predicted and recorded z and their correlation, which separates a scaling problem from a structural one. If the correct choice exists among these variants, its error should drop to near zero; if all variants stay high, the error is in the bundle encoding instead.

```python
def zpost_diag(opt, stays, act='gelu', gap_mode='log_s_10', flow=True, part='first', max_n=150):
    errs, nr, cors = [], [], []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if aa or not wa: continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            hf = h_p * torch.exp(-dt * torch.exp(T_['core.flow.a'])) if flow else h_p
            gin = {'log_s_10': math.log(dt)/10, 'log_s': math.log(dt), 'log1p_s_10': math.log1p(dt)/10, 'log_h_10': math.log(dt/3600)/10,
                   'log_min_10': math.log(dt/60)/10, 'zero': 0.0}[gap_mode]
            ctx = out_p + ln_(hf.mean(0), 'core.out_norm') + mlp_(torch.tensor([gin]), 'gap', act)
            e = setenc([atom_vec(a, opt) for a in wa], 'encode', act)
            zo = mlp_(torch.cat([ctx, e, torch.zeros(192)]), 'z_post', act)
            zh = {'first': zo[:32], 'second': zo[32:], 'even': zo[0::2], 'odd': zo[1::2]}[part]
            zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt))); nr.append(float(torch.norm(zh)/torch.norm(zt)))
            cors.append(float(np.corrcoef(zh.numpy(), zt.numpy())[0,1]))
            if len(errs) >= max_n: break
        if len(errs) >= max_n: break
    return round(np.median(errs),4), round(np.median(nr),3), round(np.median(cors),3)
optA = {'unit': {'glucose_finger':3, 'glucose_lab':3, 'map_arterial':4, 'map_cuff':4}, 'rate_unit': {}, 'side': 'left'}
for part in ['first','second','even','odd']:
    print('part', part, zpost_diag(optA, stays_t, part=part))
for gm in ['log_s_10','log_s','log1p_s_10','log_h_10','log_min_10','zero']:
    print('gap', gm, zpost_diag(optA, stays_t, gap_mode=gm), 'noflow', zpost_diag(optA, stays_t, gap_mode=gm, flow=False))
```