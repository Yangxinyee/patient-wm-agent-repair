The row norms settle most of the embedding layout. Rows near 0.28 are at initialisation scale, meaning never trained.
- **Never used:** the action, discharged_to, route, status and text tables (only the zero row 0 is used), the REVEAL kind, and kq rows 1–3.
- **Used:** unit rows 2–5 (four trained units for mmHg, mg/dL, units/hour and mcg/kg/min, in some order); rate_unit rows 2–3; kq row 0 (perhaps added to every atom); kind rows WORLD and END; ekind rows 1–4.

So an atom vector should be: code + kind + ekind + unit[u] + rate_unit[ru] (+ kq[0]?) + the value/dose/rate bins + numeric·[z_value, has_value, z_dose, has_dose, z_rate, has_rate] + bias.

**This run.** I write a clean, vectorised atom embedding, both set encoders and the z posterior. I test them against the stored z means on all bundles of the 100 selT stays, taking each bundle's previous state from the stored states. The main variants are:
- activation (gelu vs silu);
- bin side for ties (left vs right; this matters a lot for drug rates on the 0.5 grid);
- unit-row order: vocab code order vs alphabetical;
- including kq[0] or not.

A clear winner with near-zero error confirms the input side and ctx. If every variant stays at about 0.1 error, something structural is wrong instead, such as the gap scaling or how empty e/u are handled.

```python
import torch, torch.nn.functional as F, itertools
Wt = {n_: load(n_) for n_ in NAMES}
ACT = {'gelu': F.gelu, 'silu': F.silu}
def LNt(x, pre, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), Wt[pre + '.weight'], Wt[pre + '.bias'], eps)
def mlp2l(x, pre, act): return F.linear(ACT[act](F.linear(x, Wt[pre + '.0.weight'], Wt[pre + '.0.bias'])), Wt[pre + '.2.weight'], Wt[pre + '.2.bias'])
NUMV = {(d_['code'], d_['attribute']): d_ for d_ in VOCAB['numeric']}
CODEN = {c_['id']: c_['name'] for c_ in VOCAB['codes']}
# atom table for all S bundles
rows_a = []; sidS_ = S['stay_id']; minS = np.round(S['hour'] * 60).astype(int)
bkey = {(int(s_), int(m_)): i_ for i_, (s_, m_) in enumerate(zip(sidS_, minS))}
for sid_ in pd.unique(sidS_):
    A_ = stay_atoms(trs.loc[[int(sid_)]])
    A_ = A_.assign(stay_id=int(sid_)); rows_a.append(A_)
ATM = pd.concat(rows_a, ignore_index=True)
ATM['bidx'] = [bkey.get((s_, int(m_)), -1) for s_, m_ in zip(ATM.stay_id, ATM.minute)]
print('atoms', len(ATM), 'unmatched', int((ATM.bidx < 0).sum()))
def atom_feats(df, side):
    n_ = len(df); num = np.zeros((n_, 6), np.float32); vb = np.zeros(n_, int); db = np.zeros(n_, int); rb = np.zeros(n_, int)
    for i_, (c_, v_, r_, d_) in enumerate(zip(df.code.values, df.value.values, df.rate.values, df.dose.values)):
        cn = CODEN[int(c_)]
        for col_, x_, bb in ((0, v_, vb), (2, d_, db), (4, r_, rb)):
            at_ = {0: 'value', 2: 'dose', 4: 'rate'}[col_]
            if not np.isnan(x_) and (cn, at_) in NUMV:
                nv = NUMV[(cn, at_)]
                num[i_, col_] = np.clip((x_ - nv['median']) / nv['iqr'], -10, 10); num[i_, col_ + 1] = 1.0
                bb[i_] = 1 + int(np.searchsorted(nv['fine_bin_edges'], x_, side=side))
    return num, vb, db, rb
def embed_atoms(df, num, vb, db, rb, umap, rumap, use_kq):
    W_ = Wt
    x = W_['embed.code.weight'][df.code.values] + W_['embed.kind.weight'][df.kind.values] + W_['embed.ekind.weight'][df.ekind.values]
    ur = np.array([umap.get(CODEN[int(c_)], 0) for c_ in df.code.values]); rr = np.array([rumap.get(CODEN[int(c_)], 0) for c_ in df.code.values])
    x = x + W_['embed.categorical.unit.weight'][ur] + W_['embed.categorical.rate_unit.weight'][rr]
    if use_kq: x = x + W_['embed.kq.weight'][0]
    x = x + W_['embed.bins.value.weight'][vb] + W_['embed.bins.dose.weight'][db] + W_['embed.bins.rate.weight'][rb]
    x = x + torch.from_numpy(num) @ W_['embed.numeric.weight'].T + W_['embed.numeric.bias']
    return x
def bundle_enc(xa, bidx, isact, nb, act):
    outE = torch.zeros(nb, 192); outU = torch.zeros(nb, 192)
    for flag, pre, outT in ((False, 'encode', outE), (True, 'encode_action', outU)):
        m_ = isact == flag
        if m_.sum() == 0: continue
        h1 = mlp2l(xa[m_], pre + '.mlp1', act)
        bi = torch.from_numpy(bidx[m_]); ssum = torch.zeros(nb, 192).index_add_(0, bi, h1)
        cnt = torch.zeros(nb).index_add_(0, bi, torch.ones(len(bi)))
        has = cnt > 0
        pooled = ssum[has] / cnt[has].sqrt()[:, None] + F.linear(torch.log1p(cnt[has])[:, None], Wt[pre + '.size.weight'], Wt[pre + '.size.bias'])
        outT[has] = LNt(mlp2l(pooled, pre + '.mlp2', act), pre + '.norm')
    return outE, outU
xS_t = torch.from_numpy(xS.astype(np.float32)); hS_ = S['hour']; nB = len(hS_)
prev_ok = np.r_[False, sidS_[1:] == sidS_[:-1]]
ip_ = np.where(prev_ok)[0]
def z_check(act, side, umap, rumap, use_kq, gap_div=10.0, zhalf=0):
    num, vb, db, rb = atom_feats(ATM, side)
    xa = embed_atoms(ATM, num, vb, db, rb, umap, rumap, use_kq)
    isact = (ATM.ekind.values == 2)
    E_, U_ = bundle_enc(xa, ATM.bidx.values, isact, nB, act)
    out_p = xS_t[ip_ - 1, :192]; h_p = xS_t[ip_ - 1, 192:960].reshape(-1, 4, 192)
    dt_ = torch.from_numpy(((hS_[ip_] - hS_[ip_ - 1]) * 3600).astype(np.float32))
    hf = h_p * torch.exp(-dt_[:, None, None] * torch.exp(Wt['core.flow.a']))
    g_ = mlp2l((torch.log(dt_) / gap_div)[:, None], 'gap', act)
    ctx = out_p + LNt(hf.mean(1), 'core.out_norm') + g_
    zp = mlp2l(torch.cat([ctx, E_[ip_], U_[ip_]], 1), 'z_post', act)
    zm = zp[:, :32] if zhalf == 0 else zp[:, 32:]
    err = (zm - xS_t[ip_, 960:992]).abs()
    return float(err.mean()), float(err.max()), E_, U_
U_codes = {'map_cuff': 2, 'glucose_finger': 3, 'map_arterial': 2, 'glucose_lab': 3, 'insulin': 4, 'norepinephrine': 5}
U_alpha = {'norepinephrine': 2, 'glucose_finger': 3, 'glucose_lab': 3, 'map_cuff': 4, 'map_arterial': 4, 'insulin': 5}
RU1 = {'insulin': 2, 'norepinephrine': 3}; RU2 = {'insulin': 3, 'norepinephrine': 2}
for act_, side_, (unm, um), (rnm, ru), kq_ in itertools.product(['gelu', 'silu'], ['left', 'right'], [('codes', U_codes), ('alpha', U_alpha)], [('ru1', RU1), ('ru2', RU2)], [True, False]):
    me_, mx_, _, _ = z_check(act_, side_, um, ru, kq_)
    print(f'{act_:4s} {side_:5s} {unm:5s} {rnm} kq={int(kq_)}  mean|err| {me_:.4f}  max {mx_:.3f}')
```