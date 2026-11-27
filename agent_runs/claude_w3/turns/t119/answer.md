The baseline (strict "<" bins, dt in seconds, divided by 10, slot mean through out_norm) is the best of these at 0.103. The other variants are clearly worse, so those details are right.

The remaining error could come from how the posterior is parametrised rather than from its inputs. For example, the posterior mean might be the prior mean plus a correction, or the output halves might be in the other order. I'll test both, and also measure how the error is spread across bundles.

```python
def zparts(opt={}):
    rows_ = []
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
            ctx_ = out_p + ln_(hf.mean(0), 'core.out_norm', P0) + mlp_(torch.tensor([np.log(dt_)/10]), 'gap', P0)
            po = mlp_(torch.cat([ctx_, e_, u_]), 'z_post', P0); pr = mlp_(ctx_, 'z_prior', P0)
            rows_.append((po.numpy(), pr.numpy(), X4d[idx_[k_], 960:992].numpy(), dt_))
    return rows_
zr_ = zparts()
PO = np.stack([r[0] for r in zr_]); PR = np.stack([r[1] for r in zr_]); ZT = np.stack([r[2] for r in zr_])
rm = lambda a: np.sqrt(((a - ZT)**2).mean())
print('post[:32]', rm(PO[:,:32]), 'post[32:]', rm(PO[:,32:]), 'prior[:32]', rm(PR[:,:32]), 'prior+post', rm(PR[:,:32]+PO[:,:32]))
e_ = np.sqrt(((PO[:,:32]-ZT)**2).mean(1))
print('per-bundle err quantiles', np.round(np.quantile(e_, [.1,.5,.9,.99]), 3))
print('per-dim err', np.round(np.sqrt(((PO[:,:32]-ZT)**2).mean(0)), 2))
```