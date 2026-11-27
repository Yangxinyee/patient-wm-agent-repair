Routing END atoms through the measurement encoder fits far better than the alternatives: error 0.017, against 0.069 for dropping them and about 0.18-0.19 for the action encoder or both. So END atoms belong in e, but their embedding is still slightly off. Next I'll grid the parts of an END atom's embedding while routing it through e: whether unit and rate-unit rows are included, whether rate and dose are attached, the kq index, and the action and status rows. An exact zero would finish the encoder.

```python
def end_embed(A_, cfg, opts):
    code = A_.code.values; n = len(A_)
    V = Wt('embed.code.weight')[[CID[c] for c in code]] + Wt('embed.kind.weight')[2] + Wt('embed.ekind.weight')[[EKc[c] for c in code]] + Wt('embed.kq.weight')[opts['kq']]
    if opts['units']:
        V = V + Wt('embed.categorical.unit.weight')[[cfg['unit'][c] for c in code]] + Wt('embed.categorical.rate_unit.weight')[[cfg['runit'][c] for c in code]]
    num = np.zeros((n, 6))
    if opts['attrs']:
        for c in set(code):
            ix_ = np.where(code == c)[0]
            d_ = A_.dose.values[ix_]; r_ = A_.rate.values[ix_]
            num[ix_, 2] = zs_(c,'dose',d_); num[ix_, 3] = 1; num[ix_, 4] = zs_(c,'rate',r_); num[ix_, 5] = 1
            V[ix_] += Wt('embed.bins.dose.weight')[fb_(c,'dose',d_)] + Wt('embed.bins.rate.weight')[fb_(c,'rate',r_)]
    V = V + torch.tensor(num) @ Wt('embed.numeric.weight').T + Wt('embed.numeric.bias')
    for tab in ['action', 'status']:
        if opts.get(tab) is not None: V = V + Wt(f'embed.categorical.{tab}.weight')[opts[tab]]
    return V
isd_all = A3.code.isin(['insulin','norepinephrine']).values; isend_all = isd_all & (A3.kind.values == 2)
Vbase = atoms_embed_y(A3, cfgE, {'kq': 1, 'action': 1}, {}, False)
def eu_with_end(opts, act='gelu'):
    V = Vbase.clone(); V[torch.tensor(isend_all)] = end_embed(A3[isend_all], cfgE, opts)
    out_ = {}
    for nm_, msk, pre in [('e', ~isd_all | isend_all, 'encode'), ('u', isd_all & ~isend_all, 'encode_action')]:
        m_t = torch.tensor(msk); bi = torch.tensor(A3.bi.values[msk])
        H = mlp_(V[m_t], pre + '.mlp1', act)
        S = torch.zeros(NB, 192, dtype=torch.float64).index_add_(0, bi, H)
        nn_ = torch.zeros(NB, dtype=torch.float64).index_add_(0, bi, torch.ones(int(msk.sum()), dtype=torch.float64))
        S = S / torch.sqrt(nn_.clamp(min=1))[:, None] + torch.log1p(nn_)[:, None] * Wt(pre + '.size.weight')[:, 0] + Wt(pre + '.size.bias')
        out_[nm_] = LN(mlp_(S, pre + '.mlp2', act), (192,), Wt(pre + '.norm.weight'), Wt(pre + '.norm.bias'), eps=1e-5)
    return out_
res9 = []
for kqe, un_, at_, ac_, st_ in itertools.product(range(4), [True, False], [True, False], [None, 0, 1], [None, 0, 1]):
    eu = eu_with_end({'kq': kqe, 'units': un_, 'attrs': at_, 'action': ac_, 'status': st_})
    zp = mlp_(torch.cat([ctxF, eu['e'], eu['u']], 1), 'z_post', 'gelu')[:, :32]
    er_ = (zp - zm3).abs().mean(1).numpy()
    res9.append((round(er_[se_].mean(), 6), kqe, un_, at_, ac_, st_))
res9.sort(key=lambda r: r[0]); print(res9[:5])
```