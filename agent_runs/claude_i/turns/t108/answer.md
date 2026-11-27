Sending END atoms through the measurement encoder cuts the single-END error from 0.19 to 0.017, so END atoms likely don't belong in the action set, but this isn't exact either. Next I'll test the remaining routings: END atoms dropped from both encoders, or included in both. A near-zero error under "dropped" would mean the bundle encoders ignore END atoms. Their timing would then reach the model only through the bundle's time and the empty-set encodings, and END atoms would play no part in the dose pathway.

```python
def eu_routed2(cfg, ex_start, ex_end, end_attrs, end_to, act='gelu'):
    A_ = A3
    V = atoms_embed_y(A_, cfg, ex_start, ex_end, end_attrs)
    isd = A_.code.isin(['insulin','norepinephrine']).values; k_ = A_.kind.values; isend = isd & (k_ == 2)
    out_ = {}
    for nm_, pre in [('e', 'encode'), ('u', 'encode_action')]:
        if nm_ == 'e': msk = ~isd | (isend & (end_to in ('e', 'both')))
        else: msk = (isd & (k_ == 0)) | (isend & (end_to in ('u', 'both')))
        m_t = torch.tensor(msk); bi = torch.tensor(A_.bi.values[msk])
        H = mlp_(V[m_t], pre + '.mlp1', act)
        S = torch.zeros(NB, 192, dtype=torch.float64).index_add_(0, bi, H)
        n = torch.zeros(NB, dtype=torch.float64).index_add_(0, bi, torch.ones(int(msk.sum()), dtype=torch.float64))
        S = S / torch.sqrt(n.clamp(min=1))[:, None] + torch.log1p(n)[:, None] * Wt(pre + '.size.weight')[:, 0] + Wt(pre + '.size.bias')
        out_[nm_] = LN(mlp_(S, pre + '.mlp2', act), (192,), Wt(pre + '.norm.weight'), Wt(pre + '.norm.bias'), eps=1e-5)
    return out_
res8 = []
for end_to in ['drop', 'both']:
    for kqe, ace, ea in itertools.product(range(4), [None, 1], [True, False]):
        eu = eu_routed2(cfgE, {'kq': 1, 'action': 1}, {'kq': kqe, 'action': ace}, ea, end_to)
        zp = mlp_(torch.cat([ctxF, eu['e'], eu['u']], 1), 'z_post', 'gelu')[:, :32]
        er_ = (zp - zm3).abs().mean(1).numpy()
        res8.append((round(er_[se_].mean(), 6), end_to, kqe, ace, ea))
        if end_to == 'drop': break
res8.sort(key=lambda r: r[0]); print(res8[:5])
```