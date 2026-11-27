Single-END bundles have error 0.19 whatever the END atom's embedding. That suggests END atoms don't go through the action encoder at all. Perhaps they are treated as world events and go through the measurement encoder (e), leaving the action set empty. Next I'll test that routing on single-END bundles, varying the END atom's kq, action row and attributes. A near-zero error would confirm that END atoms enter e, not u. That matters for the edit: only START atoms would then carry the dose into the action pathway.

```python
def eu_routed(cfg, ex_start, ex_end, end_attrs, end_to='e', act='gelu'):
    A_ = A3
    V = atoms_embed_y(A_, cfg, ex_start, ex_end, end_attrs)
    isd = A_.code.isin(['insulin','norepinephrine']).values; k_ = A_.kind.values
    to_u = isd & ((k_ == 0) | (end_to == 'u'))
    out_ = {}
    for nm_, msk, pre in [('e', ~to_u, 'encode'), ('u', to_u, 'encode_action')]:
        m_t = torch.tensor(msk); bi = torch.tensor(A_.bi.values[msk])
        H = mlp_(V[m_t], pre + '.mlp1', act)
        S = torch.zeros(NB, 192, dtype=torch.float64).index_add_(0, bi, H)
        n = torch.zeros(NB, dtype=torch.float64).index_add_(0, bi, torch.ones(int(msk.sum()), dtype=torch.float64))
        S = S / torch.sqrt(n.clamp(min=1))[:, None] + torch.log1p(n)[:, None] * Wt(pre + '.size.weight')[:, 0] + Wt(pre + '.size.bias')
        out_[nm_] = LN(mlp_(S, pre + '.mlp2', act), (192,), Wt(pre + '.norm.weight'), Wt(pre + '.norm.bias'), eps=1e-5)
    return out_
res7 = []
for kqe, ace, ea in itertools.product(range(4), [None, 0, 1], [True, False]):
    eu = eu_routed(cfgE, {'kq': 1, 'action': 1}, {'kq': kqe, 'action': ace}, ea, end_to='e')
    zp = mlp_(torch.cat([ctxF, eu['e'], eu['u']], 1), 'z_post', 'gelu')[:, :32]
    er_ = (zp - zm3).abs().mean(1).numpy()
    res7.append((round(er_[se_].mean(), 6), kqe, ace, ea))
res7.sort(key=lambda r: r[0]); print(res7[:5])
```