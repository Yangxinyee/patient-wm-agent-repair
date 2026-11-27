Two results here. First, the prior mean alone comes almost as close to the dumped z (0.130) as my posterior does (0.107), so my bundle encoding e_k adds very little. Second, the error is spread evenly across bundles (median 0.104), so the mismatch is systematic.

Either e_k barely matters or my e_k is wrong. To tell which, I'll compare the error with e_k set to zero and with e_k taken from the wrong bundle. If those are no worse than my current e_k, then my encoder carries no correct information.

```python
def zpost_err(e_mode='own'):
    errs = []
    for s_ in stays12:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        for k_ in range(1, len(bl_)-1):
            g_ = bl_[k_]; isact = g_.code.isin([8,9]).values
            if isact.any(): continue
            gg_ = bl_[k_+1] if e_mode == 'next' else g_
            vecs = [atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, {}) for r in gg_.itertuples() if r.code not in (8,9)]
            e_ = set_enc_(vecs, 'encode', P0)
            if e_mode == 'zero': e_ = torch.zeros(192, dtype=torch.float64)
            u_ = torch.zeros(192, dtype=torch.float64)
            prev = X4d[idx_[k_-1]]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt_ = (g_.minute.iloc[0] - bl_[k_-1].minute.iloc[0])*60.0
            hf = h_p*torch.exp(-dt_*torch.exp(P0['core.flow.a']))
            ctx_ = out_p + ln_(hf.mean(0), 'core.out_norm', P0) + mlp_(torch.tensor([np.log(dt_)/10]), 'gap', P0)
            zm = mlp_(torch.cat([ctx_, e_, u_]), 'z_post', P0)[:32]
            errs.append(((zm - X4d[idx_[k_], 960:992])**2).mean().sqrt().item())
    return round(float(np.mean(errs)), 4)
for m_ in ['own', 'zero', 'next']:
    print(m_, zpost_err(m_))
```