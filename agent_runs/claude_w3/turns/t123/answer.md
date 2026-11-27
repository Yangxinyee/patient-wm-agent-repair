Rounding to bf16 moves the output by only 0.0007, so precision is not the cause. The 0.10 mismatch is systematic. e_k clearly matters, so the next suspect is the set encoder's pooling. I'll test these variants against the dump:
- mean pooling instead of sum/√n;
- plain sum;
- n instead of log(1+n) in the size term;
- no final norm.

```python
def set_enc_v(vecs, p, Pp, var):
    n_ = len(vecs)
    if n_ == 0: return torch.zeros(192, dtype=torch.float64)
    m1 = mlp_(torch.stack(vecs), p+'.mlp1', Pp)
    pool = {'sqrt': m1.sum(0)/np.sqrt(n_), 'mean': m1.mean(0), 'sum': m1.sum(0)}[var.get('pool','sqrt')]
    sz = {'log1p': np.log1p(n_), 'n': float(n_), 'log': np.log(n_)}[var.get('size','log1p')]
    s_ = pool + lin_(torch.tensor([sz], dtype=torch.float64), p+'.size', Pp)
    o_ = mlp_(s_, p+'.mlp2', Pp)
    return ln_(o_, p+'.norm', Pp) if var.get('norm', True) else o_
def err_var(var, nmax=150):
    errs = []
    for s_ in stays12:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        for k_ in range(1, len(bl_)):
            g_ = bl_[k_]
            if g_.code.isin([8,9]).any(): continue
            vecs = [atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, {}) for r in g_.itertuples()]
            e_ = set_enc_v(vecs, 'encode', P0, var); u_ = torch.zeros(192, dtype=torch.float64)
            prev = X4d[idx_[k_-1]]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt_ = (g_.minute.iloc[0] - bl_[k_-1].minute.iloc[0])*60.0
            hf = h_p*torch.exp(-dt_*torch.exp(P0['core.flow.a']))
            ctx_ = out_p + ln_(hf.mean(0), 'core.out_norm', P0) + mlp_(torch.tensor([np.log(dt_)/10]), 'gap', P0)
            zm = mlp_(torch.cat([ctx_, e_, u_]), 'z_post', P0)[:32]
            errs.append(((zm - X4d[idx_[k_], 960:992])**2).mean().sqrt().item())
    return round(float(np.mean(errs)), 4)
for var in [dict(), dict(pool='mean'), dict(pool='sum'), dict(size='n'), dict(size='log'), dict(norm=False)]:
    print(var, err_var(var))
```