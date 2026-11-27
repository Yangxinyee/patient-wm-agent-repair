No variant reproduces z. All gelu variants sit at a mean error of about 0.10 (max about 3.2) against a z scale of 0.17. Silu is clearly worse, so gelu is the activation. The bin side, unit order and kq barely matter. That points to a structural mismatch rather than an embedding detail. The candidates are:
- how ctx is formed (gap scaling, whether h is time-flowed);
- the order of z_post's inputs;
- which half of z_post's output is the mean;
- a convention in how the stored states line up with bundles.

**This run.** I keep the best embedding variant (gelu, left, code-order units, kq) and test these structural alternatives one at a time. I also break the error down by bundle type (reading-only, action-only, mixed) and by position in the stay. One variant should drop the error by an order of magnitude. If none does, the per-type breakdown will localise the problem, for example to errors only on action bundles (the action encoder or u handling) or only on reading bundles.

```python
def z_check2(act='gelu', side='left', umap=U_codes, rumap=RU1, use_kq=True, gap_mode='log10s', zhalf=0, order='ceu', flow=True, outn=True):
    num, vb, db, rb = atom_feats(ATM, side)
    xa = embed_atoms(ATM, num, vb, db, rb, umap, rumap, use_kq)
    E_, U_ = bundle_enc(xa, ATM.bidx.values, ATM.ekind.values == 2, nB, act)
    out_p = xS_t[ip_ - 1, :192]; h_p = xS_t[ip_ - 1, 192:960].reshape(-1, 4, 192)
    dt_ = torch.from_numpy(((hS_[ip_] - hS_[ip_ - 1]) * 3600).astype(np.float32))
    hf = h_p * torch.exp(-dt_[:, None, None] * torch.exp(Wt['core.flow.a'])) if flow else h_p
    gin = {'log10s': torch.log(dt_) / 10, 'logs': torch.log(dt_), 'log1p10': torch.log1p(dt_) / 10, 'logh': torch.log(dt_ / 3600), 'none': None}[gap_mode]
    g_ = mlp2l(gin[:, None], 'gap', act) if gin is not None else 0
    ctx = out_p + (LNt(hf.mean(1), 'core.out_norm') if outn else hf.mean(1)) + g_
    parts = {'c': ctx, 'e': E_[ip_], 'u': U_[ip_]}
    zp = mlp2l(torch.cat([parts[ch] for ch in order], 1), 'z_post', act)
    zm = zp[:, :32] if zhalf == 0 else zp[:, 32:]
    return (zm - xS_t[ip_, 960:992]).abs().mean(1).numpy()
base = z_check2()
print('base', base.mean().round(4))
for kw in [dict(zhalf=1), dict(gap_mode='logs'), dict(gap_mode='log1p10'), dict(gap_mode='logh'), dict(gap_mode='none'), dict(order='eu' + 'c'), dict(order='cue'), dict(flow=False), dict(outn=False)]:
    print(kw, z_check2(**kw).mean().round(4))
nat_b = S['n_atoms'][ip_]; hi_ = S['has_insulin'][ip_] | S['has_norepinephrine'][ip_]
hasmeas = np.zeros(nB, bool); hasmeas[ATM.bidx.values[ATM.ekind.values != 2]] = True
for nm_, m_ in [('meas only', hasmeas[ip_] & ~hi_), ('action only', ~hasmeas[ip_] & hi_), ('both', hasmeas[ip_] & hi_)]:
    print(nm_, int(m_.sum()), 'mean err', base[m_].mean().round(4), 'median', np.median(base[m_]).round(4))
pos = np.array([np.sum(sidS_[:i_] == sidS_[i_]) for i_ in ip_])
print('err by position (1,2,3,5,10,20+):', [base[pos == p_].mean().round(3) if (pos == p_).any() else None for p_ in [1, 2, 3, 5, 10]], base[pos >= 20].mean().round(3))
```