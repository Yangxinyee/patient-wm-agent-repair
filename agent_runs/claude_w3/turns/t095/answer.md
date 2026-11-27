The z-posterior check failed for every activation and unit mapping. The typical max-abs error is 0.35, far above what a near-miss in the embedding would produce, so the mistake is structural.

The most likely suspects:
- the dt scale inside `gap` (dt in seconds or hours, divided by 10 or not);
- whether the z mean is the first or second half of z_post's output;
- whether h gets the time flow before entering the context.

I will test these variants with GELU and the best unit mapping. One variant dropping to ~1e-5 identifies the error. If none moves much, the encoder output itself is the problem.

```python
def zpost_check2(opts, stays_=None, only_meas=True):
    errs = []; X4 = torch.tensor(F4['x'], dtype=torch.float64); fa = torch.exp(Tt['core.flow.a'])
    for s_ in (stays_ or stays12):
        idx_ = np.where(F4['stay_id'] == s_)[0]; A_ = BUN[s_]; mins = sorted(A_.minute.unique())
        for k_ in range(1, len(mins)):
            rows = A_[A_.minute == mins[k_]]
            if only_meas and (rows.ekind != 1).any(): continue
            e_, u_, _, _ = bundle_vecs(rows)
            if opts.get('zero_e'): e_ = e_*0
            prev = X4[idx_[k_-1]]; dt_ = 60.0*(mins[k_]-mins[k_-1])
            hprev = prev[192:960].reshape(4, 192)
            if opts.get('flow', True): hprev = hprev*torch.exp(-dt_*fa)
            g_in = {'s10': np.log(dt_)/10, 's1': np.log(dt_), 'h10': np.log(dt_/3600)/10, 'm10': np.log(dt_/60)/10, 's1p10': np.log1p(dt_)/10}[opts.get('gap','s10')]
            ctx = prev[:192] + LN(hprev.mean(0), 'core.out_norm') + mlp(torch.tensor([g_in], dtype=torch.float64), 'gap')
            zz = lin(ACT[CFG['act']](lin(torch.cat([ctx, e_, u_]), 'z_post.0')), 'z_post.2')
            zz = zz[32:] if opts.get('second') else zz[:32]
            errs.append(float((zz - X4[idx_[k_], 960:992]).abs().max()))
    return round(float(np.median(errs)),4)
CFG.update(act='gelu', unit={'glucose_finger': 3, 'glucose_lab': 3, 'map_arterial': 2, 'map_cuff': 2})
sub3 = stays12[:3]
for o_ in [{}, {'second':1}, {'gap':'s1'}, {'gap':'h10'}, {'gap':'m10'}, {'gap':'s1p10'}, {'flow':False}, {'zero_e':1}]:
    print(o_, zpost_check2(o_, sub3))
```