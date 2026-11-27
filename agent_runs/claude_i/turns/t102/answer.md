The drug numeric scales are plausible (insulin rate median 1.5, IQR 1.0; norepinephrine rate median 0.46, IQR 0.59). There are 115 bundles holding a single START atom (65 insulin, 50 norepinephrine), a clean test set. Next I'll grid the extra terms a drug atom might add:
- a separate kq index for drugs (0-3);
- each of the action, route, status and text tables, at row 0, row 1 or absent.

An exact zero on the single-START bundles would identify the drug atom's embedding.

```python
def atoms_embed_x(A_, cfg, extra):
    V = atoms_embed(A_, cfg)
    isd = torch.tensor(A_.code.isin(['insulin','norepinephrine']).values)
    for tab, idx in extra.items():
        if idx is None: continue
        if tab == 'kq':
            V[isd] += Wt('embed.kq.weight')[idx] - Wt('embed.kq.weight')[cfg.get('kq', 0)]
        else:
            nm_ = 'embed.text.weight' if tab == 'text' else f'embed.categorical.{tab}.weight'
            V[isd] += Wt(nm_)[idx]
    return V
def u_only_x(cfg, extra, A_=None, act='gelu'):
    A_ = Ad if A_ is None else A_
    V = atoms_embed_x(A_, cfg, extra); bi = torch.tensor(A_.bi.values)
    H = mlp_(V, 'encode_action.mlp1', act)
    S = torch.zeros(NB, 192, dtype=torch.float64).index_add_(0, bi, H)
    n = torch.zeros(NB, dtype=torch.float64).index_add_(0, bi, torch.ones(len(A_), dtype=torch.float64))
    S = S / torch.sqrt(n.clamp(min=1))[:, None] + torch.log1p(n)[:, None] * Wt('encode_action.size.weight')[:, 0] + Wt('encode_action.size.bias')
    return LN(mlp_(S, 'encode_action.mlp2', act), (192,), Wt('encode_action.norm.weight'), Wt('encode_action.norm.bias'), eps=1e-5)
import itertools
cfgD = {**cfgC, 'end_attrs': False}
res4 = []
for kqd, ac, ro, stt, tx in itertools.product(range(4), [None,0,1], [None,0,1], [None,0,1], [None,0,1]):
    u_ = u_only_x(cfgD, {'kq': kqd, 'action': ac, 'route': ro, 'status': stt, 'text': tx})
    zp = mlp_(torch.cat([ctxF, encC['e'], u_], 1), 'z_post', 'gelu')[:, :32]
    er_ = (zp - zm3).abs().mean(1).numpy()
    res4.append((round(er_[so].mean(), 5), kqd, ac, ro, stt, tx))
res4.sort(); print(res4[:6])
```