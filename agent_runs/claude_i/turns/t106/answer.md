The START-atom fix holds, but bundles with END atoms are still off by about 0.09, including bundles that hold a lone END atom. So END atoms are embedded differently from START atoms: perhaps another kq index, another action row, a status row, or with or without rate and dose. Next I'll grid those END-atom options on single-END bundles. An exact zero there, followed by checking all drug bundles, would complete the encoder.

```python
def atoms_embed_y(A_, cfg, ex_start, ex_end, end_attrs):
    cfg2 = {**cfg, 'end_attrs': end_attrs}
    V = atoms_embed(A_, cfg2)
    isd = A_.code.isin(['insulin','norepinephrine']).values; k_ = A_.kind.values
    for msk, ex in [(isd & (k_ == 0), ex_start), (isd & (k_ == 2), ex_end)]:
        m_t = torch.tensor(msk)
        for tab, idx in ex.items():
            if idx is None: continue
            if tab == 'kq': V[m_t] += Wt('embed.kq.weight')[idx] - Wt('embed.kq.weight')[cfg.get('kq', 0)]
            else:
                nm_ = 'embed.text.weight' if tab == 'text' else f'embed.categorical.{tab}.weight'
                V[m_t] += Wt(nm_)[idx]
    return V
def u_from_V(V, A_, act='gelu'):
    bi = torch.tensor(A_.bi.values)
    H = mlp_(V, 'encode_action.mlp1', act)
    S = torch.zeros(NB, 192, dtype=torch.float64).index_add_(0, bi, H)
    n = torch.zeros(NB, dtype=torch.float64).index_add_(0, bi, torch.ones(len(A_), dtype=torch.float64))
    S = S / torch.sqrt(n.clamp(min=1))[:, None] + torch.log1p(n)[:, None] * Wt('encode_action.size.weight')[:, 0] + Wt('encode_action.size.bias')
    return LN(mlp_(S, 'encode_action.mlp2', act), (192,), Wt('encode_action.norm.weight'), Wt('encode_action.norm.bias'), eps=1e-5)
single_end = np.zeros(NB, bool)
single_end[cnt.index[(cnt.values==1) & (kin.values==2)]] = True
se_ = single_end & prev_ok.numpy() & (encC['ne'] == 0).numpy()
print('single END bundles', se_.sum())
res6 = []
for kqe, ace, ste, ea in itertools.product(range(4), [None,0,1], [None,0,1], [True, False]):
    V = atoms_embed_y(Ad, cfgE, {'kq': 1, 'action': 1}, {'kq': kqe, 'action': ace, 'status': ste}, ea)
    u_ = u_from_V(V, Ad)
    zp = mlp_(torch.cat([ctxF, encC['e'], u_], 1), 'z_post', 'gelu')[:, :32]
    er_ = (zp - zm3).abs().mean(1).numpy()
    res6.append((round(er_[se_].mean(), 6), kqe, ace, ste, ea))
res6.sort(key=lambda r: r[0]); print(res6[:6])
```