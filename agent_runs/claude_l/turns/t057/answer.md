The error is spread almost evenly over every bundle type: mean about 0.04–0.08, with no code standing out. It grows only mildly with bundle size, and changing the pooling rule or dropping `kq` hardly helps (sum pooling is marginally better, 0.0547).

A defect shared by all bundles points to the context vector or the posterior parameterisation, rather than to the atom features. So I'll vary the parts of ctx one at a time:
- the h term with or without the time-flow decay, or dropped;
- the `out` term dropped;
- the gap term dropped;
- the posterior as the raw `z_post` output, or as the prior mean plus `z_post`.

For each variant I re-search the `gap` and `z_post` activations. A variant reaching about 1e-4 identifies the missing piece. If none improves, the encoder MLP or the LayerNorm conventions are the suspects.

```python
E1g = run_combo('gelu', 'relu', 'gelu', 'relu', True, 'sqrt')[1]
hm_first = hS[first].double().mean(1)
on_w64 = T64('core.out_norm.weight')
variants = {
 'full_flow': outS[first].double() + F.layer_norm(hm_first * torch.exp(-torch.tensor(dts)[:, None] * fa64[None]), (192,), on_w64, onb, 1e-5),
 'full_noflow': outS[first].double() + F.layer_norm(hm_first, (192,), on_w64, onb, 1e-5),
 'out_only': outS[first].double(),
 'h_only_flow': F.layer_norm(hm_first * torch.exp(-torch.tensor(dts)[:, None] * fa64[None]), (192,), on_w64, onb, 1e-5),
}
out_r = []
for vn, cb in variants.items():
    for gan in ['gelu', 'silu', 'relu', 'none']:
        for gvn in ['log_s/10', 'log_s']:
            gv_ = torch.tensor(np.log(dts)/10 if gvn == 'log_s/10' else np.log(dts))[:, None]
            ctx_ = cb + (mlp2l(gv_, 'gap', ACT[gan]) if gan != 'none' else 0)
            for zn in ['relu', 'gelu', 'silu']:
                zm = mlp2l(torch.cat([ctx_, E1g, torch.zeros_like(E1g)], 1), 'z_post', ACT[zn])[:, :32]
                zp = mlp2l(ctx_, 'z_prior', ACT[zn])[:, :32]
                for pv, zz_ in [('post', zm), ('prior+post', zm + zp)]:
                    er = (zz_ - z1_true).abs(); out_r.append((er.mean().item(), er.max().item(), vn, gan, gvn, zn, pv))
out_r.sort(key=lambda r: r[0])
for r_ in out_r[:10]: print(np.round(r_[:2], 5), r_[2:])
```