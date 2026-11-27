Switching to GELU with the saved unit map brought the error from 0.24 to 0.10. The approximate and exact GELU give the same result. That is still above the 0.08 I reached before, so another detail is off.

Candidates:
- the bin boundary convention ("edges below x": strict < or ≤);
- the scaling and units of the gap input;
- whether the slot mean goes through out_norm.

I'll test these one at a time on the same 323 measurement-only bundles.

```python
CUR_ACT = 'gelu'
def zb(code, attr, x, side=None):
    t_ = numtab.get((cname[code], attr))
    if t_ is None or x is None or np.isnan(x): return 0.0, 0.0, 0
    z_ = float(np.clip((x - t_['median'])/t_['iqr'], -10, 10))
    b_ = 1 + int(np.searchsorted(np.array(t_['fine_bin_edges']), x, side=SIDE))
    return z_, 1.0, b_
def check2(opt):
    errs = []
    for s_ in stays12:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        for k_ in range(1, len(bl_)):
            g_ = bl_[k_]; isact = g_.code.isin([8,9]).values
            if isact.any(): continue
            vecs = [atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, opt) for r in g_.itertuples()]
            e_ = set_enc_(vecs, 'encode', P0); u_ = torch.zeros(192, dtype=torch.float64)
            prev = X4d[idx_[k_-1]]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt_ = (g_.minute.iloc[0] - bl_[k_-1].minute.iloc[0])*60.0
            hf = h_p*torch.exp(-dt_*torch.exp(P0['core.flow.a']))
            gin = np.log(dt_/opt.get('du',1.0))/opt.get('gs',10.0)
            ctx_ = out_p + (ln_(hf.mean(0), 'core.out_norm', P0) if opt.get('hn',True) else 0) + mlp_(torch.tensor([gin]), 'gap', P0)
            zm = mlp_(torch.cat([ctx_, e_, u_]), 'z_post', P0)[:32]
            errs.append(((zm - X4d[idx_[k_], 960:992])**2).mean().sqrt().item())
    return round(float(np.mean(errs)),4)
for SIDE in ['left','right']:
    print('side', SIDE, check2({}))
SIDE='left'
for o_ in [dict(gs=1.0), dict(du=60.0), dict(du=3600.0), dict(hn=False)]:
    print(o_, check2(o_))
```