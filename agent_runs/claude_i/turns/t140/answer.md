The old helper's gap vector matches a ReLU gap MLP exactly. It also tracks the tool better than my GELU head (1.86 vs 4.06 mg/dL), and using sampled z did not change mine. So the heads may use ReLU where the encoders and core use GELU. Possibly the old helper simply defaulted to ReLU everywhere, through `mlp2(kind="relu")` in head.py, and that happened to be right for the heads.

Next I will run my head with ReLU or GELU chosen separately in the gap MLP, the z-prior MLP and the value-bin MLP, and keep the combination with the lowest error. The best one should beat 1.86. If it does, the head is reproduced and the remaining error is just the tool's 4-draw noise.

```python
def pred_mean_v(r_, acts, typ_=1, ns=32, seed=0, lay='A'):
    kb, ev_, obs_, pm_, pv_, pa_ = r_
    A_ = {'gelu': Fnn.gelu, 'relu': torch.relu}
    hf = HHt[kb-1] * torch.exp(-dts[kb]*torch.exp(WT['core.flow.a']))
    g_ = linw(A_[acts[0]](linw(torch.log(dts[kb]).reshape(1)/10., 'gap.0')), 'gap.2')
    ctx = OUTt[kb-1] + LNw(hf.mean(0), 'core.out_norm') + g_
    p_ = linw(A_[acts[1]](linw(ctx, 'z_prior.0')), 'z_prior.2'); zm_, zls_ = p_[:32], p_[32:]
    g = torch.Generator().manual_seed(seed); zs = zm_ + torch.exp(zls_) * torch.randn(ns, 32, generator=g)
    q = LNw(ctx + linw(zs, 'z_head') + WT['type_emb.weight'][typ_], 'query_norm')
    has = 0. if np.isnan(pv_) else 1.
    zl = 0. if not has else float(zsc(ev_, 'value', pv_)); ag = 0. if not has else np.log1p(pa_)/10.
    feats = {'A': [has, ag, zl, has, 0, 0, 0, 0], 'B': [has, ag, zl, 0, 0, 0, 0, 0]}[lay]
    lb = 0 if not has else int(fbin(ev_, 'value', np.array([pv_]))[0])
    wc = q + WT['embed.code.weight'][CID[ev_]] + torch.tensor(feats, dtype=torch.float32) @ WT['last_numeric.weight'].T + WT['last_numeric.bias'] \
         + WT['embed.bins.value.weight'][lb] + WT['embed.bins.rate.weight'][0] + WT['last_status.weight'][0]
    lg = linw(A_[acts[2]](linw(wc, 'attr_heads.value_bin.0')), 'attr_heads.value_bin.2') + WT['embed_transition.value_bin.weight'][CID[ev_]*17 + lb]
    pp = torch.softmax(lg, -1).mean(0).numpy()
    return float((pp[1:] * CENT[ev_]).sum() / pp[1:].sum())
import itertools
for acts in itertools.product(['gelu','relu'], repeat=3):
    print(acts, round(np.mean([abs(pred_mean_v(r_, acts) - r_[3]) for r_ in sub_r]), 3))
```