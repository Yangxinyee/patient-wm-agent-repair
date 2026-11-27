No gap-input variant matches the old head's gap vector; log(dt)/10, my current choice, is still the closest (difference 0.31). So the old helper probably used a different activation or form internally, but it still tracked the tool to about 1.3–1.9 mg/dL. My z-posterior check, which was exact, used GELU with log(dt)/10. So my gap is likely the correct one, and my head's larger error (4.0) comes from somewhere else.

The most likely source is z. The tool averages the bin probabilities over 4 draws from the z prior, while I plugged in the prior mean. Because the head is nonlinear in z, that can bias the predicted mean. Next I will recompute my head's error with probabilities averaged over 32 prior draws. In the same run I check whether the old helper's gap matches a ReLU-activated gap MLP, to learn what it did.
- **If sampling brings my error to about 2 or below:** the remaining structure is right.

```python
for act_nm, actfn in [('relu', torch.relu), ('silu', Fnn.silu)]:
    g_ = linw(actfn(linw(torch.tensor([np.log(dt0)/10], dtype=torch.float32), 'gap.0')), 'gap.2').numpy()
    print('gap with', act_nm, round(float(np.abs(gv_[i0] - g_).max()), 5))
def pred_mean_s(r_, typ_=1, lay='A', ns=32, seed=0, status=0, ratebin0=True):
    kb, ev_, obs_, pm_, pv_, pa_ = r_
    ctx = view_ctx(kb-1, dts[kb]); zm_, zls_ = zprior(ctx)
    g = torch.Generator().manual_seed(seed); zs = zm_ + torch.exp(zls_) * torch.randn(ns, 32, generator=g)
    view = ctx + linw(zs, 'z_head')
    q = LNw(view + WT['type_emb.weight'][typ_], 'query_norm')
    has = 0. if np.isnan(pv_) else 1.
    zl = 0. if not has else float(zsc(ev_, 'value', pv_)); ag = 0. if not has else np.log1p(pa_)/10.
    feats = {'A': [has, ag, zl, has, 0, 0, 0, 0], 'B': [has, ag, zl, 0, 0, 0, 0, 0]}[lay]
    lb = 0 if not has else int(fbin(ev_, 'value', np.array([pv_]))[0])
    wc = q + WT['embed.code.weight'][CID[ev_]] + torch.tensor(feats, dtype=torch.float32) @ WT['last_numeric.weight'].T + WT['last_numeric.bias'] \
         + WT['embed.bins.value.weight'][lb] + (WT['embed.bins.rate.weight'][0] if ratebin0 else 0) + WT['last_status.weight'][status]
    lg = linw(Fnn.gelu(linw(wc, 'attr_heads.value_bin.0')), 'attr_heads.value_bin.2') + WT['embed_transition.value_bin.weight'][CID[ev_]*17 + lb]
    p_ = torch.softmax(lg, -1).mean(0).numpy()
    return float((p_[1:] * CENT[ev_]).sum() / p_[1:].sum()), p_
print('MAD sampled z, A:', round(np.mean([abs(pred_mean_s(r_)[0] - r_[3]) for r_ in sub_r]), 3))
print('MAD sampled z, A, no ratebin0:', round(np.mean([abs(pred_mean_s(r_, ratebin0=False)[0] - r_[3]) for r_ in sub_r]), 3))
```