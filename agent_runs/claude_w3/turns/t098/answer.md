My z-posterior is partly right: correlation 0.79, slope about 1, RMSE 0.10 against a true SD of 0.16. The first half of z_post's output is indeed the mean. So the structure is roughly correct, but one input is off: either the context or the measurement encoding.

Next I will vary the context terms one at a time: drop `gap`, drop the normalised-h term, drop out, or use a different dt convention. The goal is to see which version brings the RMSE down sharply. If none does, the error lies in the encoder input e_k, and I will turn to the embedding next.

```python
def zerr(ctx_fn, stays_=sub3, e_fn=None):
    errs = []
    for s_ in stays_:
        idx_ = np.where(F4['stay_id'] == s_)[0]; A_ = BUN[s_]; mins = sorted(A_.minute.unique())
        for k_ in range(1, len(mins)):
            rows = A_[A_.minute == mins[k_]]
            if (rows.ekind != 1).any(): continue
            e_, u_, _, _ = bundle_vecs(rows)
            if e_fn: e_ = e_fn(e_)
            prev = X4[idx_[k_-1]]; dt_ = 60.0*(mins[k_]-mins[k_-1])
            ctx = ctx_fn(prev, dt_)
            zz = mlpm(torch.cat([ctx, e_, u_]), 'z_post','zp')[:32]
            errs.append(float(((zz - X4[idx_[k_], 960:992])**2).mean()))
    return round(float(np.sqrt(np.mean(errs))),4)
G_ = lambda dt_: mlpm(torch.tensor([np.log(dt_)/10], dtype=torch.float64), 'gap','gap')
HN = lambda prev, dt_, fl=True: LN((prev[192:960].reshape(4,192)*(torch.exp(-dt_*FA) if fl else 1)).mean(0), 'core.out_norm')
variants = {
 'full': lambda p, d: p[:192] + HN(p, d) + G_(d),
 'no_gap': lambda p, d: p[:192] + HN(p, d),
 'no_hn': lambda p, d: p[:192] + G_(d),
 'no_out': lambda p, d: HN(p, d) + G_(d),
 'noflow': lambda p, d: p[:192] + HN(p, d, False) + G_(d),
 'gap_min': lambda p, d: p[:192] + HN(p, d) + mlpm(torch.tensor([np.log(d/60)/10], dtype=torch.float64), 'gap','gap'),
 'flow_h': lambda p, d: p[:192] + LN((p[192:960].reshape(4,192)*torch.exp(-(d/3600)*FA)).mean(0),'core.out_norm') + G_(d),
}
for k2, f2 in variants.items(): print(k2, zerr(f2))
```