The last-value register matches exactly: same values and same ages. So the gap between my head (4.06 mg/dL from the tool) and the old helper (1.86) lies in how the head combines its inputs: the layout of the 8 last-value features, the type row, the activations, or how z is used.

The old helper is deterministic. When it gets no z samples, it presumably uses the prior mean, and it is better than mine. So I will try to reproduce its probabilities exactly with my code, using a ReLU gap and the prior-mean z, over the type rows, activations and both feature layouts.
- **An exact match:** shows its full structure.
- **Next:** I can then decide whether its remaining 1.86 is just the tool's sampling noise, or whether the tool differs again.

```python
rows_chk = [r_ for r_ in sub_r if (int(S3[r_[0]]), round(float(H3[r_[0]]),3), r_[1]) in key2i][:30]
idx_chk = [key2i[(int(S3[r_[0]]), round(float(H3[r_[0]]),3), r_[1])] for r_ in rows_chk]
def probs_det(r_, acts, typ_, lay, gapact='relu'):
    kb, ev_, obs_, pm_, pv_, pa_ = r_
    A_ = {'gelu': Fnn.gelu, 'relu': torch.relu}
    hf = HHt[kb-1] * torch.exp(-dts[kb]*torch.exp(WT['core.flow.a']))
    g_ = linw(A_[gapact](linw(torch.log(dts[kb]).reshape(1)/10., 'gap.0')), 'gap.2')
    ctx = OUTt[kb-1] + LNw(hf.mean(0), 'core.out_norm') + g_
    p_ = linw(A_[acts[0]](linw(ctx, 'z_prior.0')), 'z_prior.2')
    q = LNw(ctx + linw(p_[:32], 'z_head') + WT['type_emb.weight'][typ_], 'query_norm')
    has = 0. if np.isnan(pv_) else 1.
    zl = 0. if not has else float(zsc(ev_, 'value', pv_)); ag = 0. if not has else np.log1p(pa_)/10.
    feats = {'A': [has, ag, zl, has, 0, 0, 0, 0], 'B': [has, ag, zl, 0, 0, 0, 0, 0]}[lay]
    lb = 0 if not has else int(fbin(ev_, 'value', np.array([pv_]))[0])
    wc = q + WT['embed.code.weight'][CID[ev_]] + torch.tensor(feats, dtype=torch.float32) @ WT['last_numeric.weight'].T + WT['last_numeric.bias'] \
         + WT['embed.bins.value.weight'][lb] + WT['embed.bins.rate.weight'][0] + WT['last_status.weight'][0]
    lg = linw(A_[acts[1]](linw(wc, 'attr_heads.value_bin.0')), 'attr_heads.value_bin.2') + WT['embed_transition.value_bin.weight'][CID[ev_]*17 + lb]
    return torch.softmax(lg, -1).numpy()
best_ = []
for acts in itertools.product(['gelu','relu'], repeat=2):
    for lay in ['A','B']:
        for t_ in range(15):
            d_ = np.mean([np.abs(probs_det(r_, acts, t_, lay) - prv[i_]).max() for r_, i_ in zip(rows_chk[:10], idx_chk[:10])])
            best_.append((d_, acts, lay, t_))
best_.sort(key=lambda x: x[0]); print([(round(b[0],4), b[1], b[2], b[3]) for b in best_[:5]])
```