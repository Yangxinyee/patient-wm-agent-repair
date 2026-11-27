These baselines are informative:
- Predicting the mean z gives an error of 0.083 and mine is 0.056, so the reimplementation is only modestly better than trivial.
- Shuffling e across stays raises the error to 0.104, so my e carries real information.
- Shuffling ctx does nothing. That is expected here, because every stay has the same state after the identical first bundle, so this test cannot assess ctx.
- Replacing the zero action vector u with the encoding of an empty action set, `norm(mlp2(size(log 1)))`, drops the error to 0.045.

So "u is zero when there are no actions" is probably wrong: the model seems to encode the empty set. Next I'll search jointly over:
- the measurement encoder activations;
- the activation used in the empty-set action encoding;
- the `z_post` activation.

A near-zero error would confirm the empty-set convention. Then I will apply the same idea to e in drug-only bundles.

```python
def empty_enc(pre, a2):
    s_ = F.linear(torch.tensor([0.0], dtype=torch.float64), T64(pre+'.size.weight'), T64(pre+'.size.bias'))
    return F.layer_norm(mlp2l(s_, pre+'.mlp2', a2), (192,), T64(pre+'.norm.weight'), T64(pre+'.norm.bias'), 1e-5)
rs = []
for a1n in ['gelu', 'relu', 'silu']:
    for a2n in ['gelu', 'relu', 'silu']:
        for pool in ['sqrt']:
            E_ = torch.stack([setenc_v(aa, 'encode', ACT[a1n], ACT[a2n], False, False, pool) for aa in atoms1])
            for ua in ['gelu', 'relu', 'silu']:
                U_ = empty_enc('encode_action', ACT[ua])[None].expand_as(E_)
                for gan in ['gelu', 'silu']:
                    ctx_ = variants['full_flow'] + mlp2l(torch.tensor(np.log(dts)/10)[:, None], 'gap', ACT[gan])
                    for zn in ['relu', 'gelu', 'silu']:
                        zz_ = mlp2l(torch.cat([ctx_, E_, U_], 1), 'z_post', ACT[zn])[:, :32]
                        er = (zz_ - z1_true).abs(); rs.append((er.mean().item(), er.max().item(), a1n, a2n, ua, gan, zn))
rs.sort(key=lambda r: r[0])
for r_ in rs[:8]: print(np.round(r_[:2], 6), r_[2:])
```