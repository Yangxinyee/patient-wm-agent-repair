The encoder for single atoms is defined. Next comes the set encoder, run per bundle and separately for measurement atoms and drug atoms, followed by the z-posterior check. ctx is built from the tool's previous state:
- the reported out and h of the previous bundle;
- h decayed over dt seconds;
- the gap MLP applied to the scaled log dt.

I'll also check whether row 0 of the bin and unit tables is zero (padding), which decides whether absent fields add anything. I'll report the z error for measurement-only and action-only bundles for a few activation and gap conventions, using frequency-order units (cfg0) as the base. An error far below the z spread (0.26) for one convention would identify it.

```python
for nm_ in ['embed.bins.value.weight','embed.bins.dose.weight','embed.categorical.unit.weight','embed.categorical.rate_unit.weight','embed.code.weight']:
    print(nm_, 'row0 norm', round(Wt(nm_)[0].norm().item(), 4), 'row1', round(Wt(nm_)[1].norm().item(), 3))
A3['bkey'] = A3.stay_id.astype(str) + '_' + A3.minute.astype(str)
tool['bkey'] = tool.stay_id.astype(str) + '_' + tool.minute.astype(str)
bpos = {k: i for i, k in enumerate(tool.bkey)}
A3['bi'] = A3.bkey.map(bpos)
NB = len(tool)
def encode_bundles(A_, cfg, act, nb=NB, empty='zero'):
    V = atoms_embed(A_, cfg); bi = torch.tensor(A_.bi.values)
    isd = torch.tensor(A_.code.isin(['insulin','norepinephrine']).values)
    res = {}
    for nm_, mask, pre in [('e', ~isd, 'encode'), ('u', isd, 'encode_action')]:
        H = mlp_(V[mask], pre + '.mlp1', act)
        S = torch.zeros(nb, 192, dtype=torch.float64).index_add_(0, bi[mask], H)
        n = torch.zeros(nb, dtype=torch.float64).index_add_(0, bi[mask], torch.ones(int(mask.sum()), dtype=torch.float64))
        S = S / torch.sqrt(n.clamp(min=1))[:, None] + torch.log1p(n)[:, None] * Wt(pre + '.size.weight')[:, 0] + Wt(pre + '.size.bias')
        O = LN(mlp_(S, pre + '.mlp2', act), (192,), Wt(pre + '.norm.weight'), Wt(pre + '.norm.bias'), eps=1e-5)
        if empty == 'zero': O[n == 0] = 0
        res[nm_] = O; res['n' + nm_] = n
    return res
min_ = torch.tensor(tool.minute.values, dtype=torch.float64)
prev_ok = torch.tensor(np.r_[False, tool.stay_id.values[1:] == tool.stay_id.values[:-1]])
dt_s = torch.zeros(NB, dtype=torch.float64); dt_s[1:] = (min_[1:] - min_[:-1]) * 60
def ctx_of(gscale, act):
    hp = torch.zeros(NB, 4, 192, dtype=torch.float64); op = torch.zeros(NB, 192, dtype=torch.float64)
    hp[1:] = h3[:-1]; op[1:] = out3[:-1]
    hf = hp * torch.exp(-dt_s[:, None, None] * torch.exp(Wt('core.flow.a')))
    g_in = gscale(dt_s.clamp(min=1))[:, None]
    return op + LN(hf.mean(1), (192,), Wt('core.out_norm.weight'), Wt('core.out_norm.bias'), eps=1e-5) + mlp_(g_in, 'gap', act)
gs = {'ls10': lambda d: torch.log(d)/10, 'ls': lambda d: torch.log(d), 'l1p10': lambda d: torch.log1p(d)/10, 'lh10': lambda d: torch.log(d/3600)/10}
for act in ['relu','gelu']:
    enc_ = encode_bundles(A3, cfg0, act)
    for gk, gf_ in gs.items():
        ctx = ctx_of(gf_, act)
        zp = mlp_(torch.cat([ctx, enc_['e'], enc_['u']], 1), 'z_post', act)[:, :32]
        err = (zp - zm3).abs().mean(1)
        mo = prev_ok & (enc_['nu'] == 0); ao = prev_ok & (enc_['ne'] == 0)
        print(act, gk, 'meas-only', round(err[mo].mean().item(), 4), 'act-only', round(err[ao].mean().item(), 4))
```