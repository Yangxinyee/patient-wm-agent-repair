All 12 stays now match exactly in bundle times and atom counts, so the input construction is right.

Next I will build the atom embedding, the two DeepSets encoders, the context vector and the z-posterior. Then I will compare my z-posterior means with the returned ones on bundles that hold only measurements. Those bundles depend only on the measurement units, not on the drug-unit mapping or END-atom details. I will search over the activation function (GELU, tanh-GELU, ReLU, SiLU) and over which unit rows mean mg/dL and mmHg. One combination with near-zero error would confirm the embedding, the encoder and the z_post structure. Errors that stay large for every combination would mean a structural misunderstanding, such as how dt enters `gap`.

```python
Tt = {n_: load_original(n_).double() for n_ in NAMES}
F = torch.nn.functional
ACT = {'gelu': F.gelu, 'gelu_t': lambda x: F.gelu(x, approximate='tanh'), 'relu': F.relu, 'silu': F.silu}
NUM = {(e_['code'], e_['attribute']): (e_['median'], e_['iqr'], np.array(e_['fine_bin_edges'])) for e_ in VOCAB['numeric']}
inv_code = {v_: k_ for k_, v_ in code_id.items()}
def lin(x, p, bias=True): 
    y = x @ Tt[p+'.weight'].T
    return y + Tt[p+'.bias'] if bias and (p+'.bias') in Tt else y
def LN(x, p, eps=1e-5): return F.layer_norm(x, (x.shape[-1],), Tt[p+'.weight'], Tt[p+'.bias'], eps)
def zs(c_, a_, x): m_, i_, _ = NUM[(c_, a_)]; return float(np.clip((x-m_)/i_, -10, 10))
def fb(c_, a_, x): return int(1 + np.sum(NUM[(c_, a_)][2] < x))
CFG = dict(act='gelu', unit={}, rate_unit={}, end_attrs=True)
def atom_emb(c_, k_, ek_, val, rate, dose):
    cn = inv_code[c_]; v_ = Tt['embed.code.weight'][c_] + Tt['embed.kind.weight'][k_] + Tt['embed.ekind.weight'][ek_] + Tt['embed.kq.weight'][0]
    v_ = v_ + Tt['embed.categorical.unit.weight'][CFG['unit'].get(cn, 0)] + Tt['embed.categorical.rate_unit.weight'][CFG['rate_unit'].get(cn, 0)]
    num = torch.zeros(6, dtype=torch.float64)
    if not np.isnan(val):
        num[0] = zs(cn, 'value', val); num[1] = 1; v_ = v_ + Tt['embed.bins.value.weight'][fb(cn, 'value', val)]
    if (not np.isnan(dose)) and (k_ == 0 or CFG['end_attrs']):
        num[2] = zs(cn, 'dose', dose); num[3] = 1; num[4] = zs(cn, 'rate', rate); num[5] = 1
        v_ = v_ + Tt['embed.bins.dose.weight'][fb(cn, 'dose', dose)] + Tt['embed.bins.rate.weight'][fb(cn, 'rate', rate)]
    return v_ + Tt['embed.numeric.weight'] @ num + Tt['embed.numeric.bias']
def mlp(x, p): return lin(ACT[CFG['act']](lin(x, p+'.0')), p+'.2')
def set_enc(embs, p):
    if len(embs) == 0: return torch.zeros(192, dtype=torch.float64)
    X_ = torch.stack(embs); n_ = X_.shape[0]
    s_ = mlp(X_, p+'.mlp1').sum(0)/np.sqrt(n_) + lin(torch.tensor([np.log1p(n_)], dtype=torch.float64), p+'.size')
    return LN(mlp(s_, p+'.mlp2'), p+'.norm')
def bundle_vecs(rows):
    em, ac = [], []
    for r_ in rows.itertuples(index=False):
        v_ = atom_emb(r_.code, r_.kind, r_.ekind, r_.value, r_.rate, r_.dose)
        (ac if r_.ekind == 2 else em).append(v_)
    return set_enc(em, 'encode'), set_enc(ac, 'encode_action'), len(em) > 0, len(ac) > 0
def zpost_check(only_meas=True, gap_div=10.0):
    errs = []
    X4 = torch.tensor(F4['x'], dtype=torch.float64)
    fa = torch.exp(Tt['core.flow.a'])
    for s_ in stays12:
        idx_ = np.where(F4['stay_id'] == s_)[0]; A_ = BUN[s_]; mins = sorted(A_.minute.unique())
        for k_ in range(1, len(mins)):
            rows = A_[A_.minute == mins[k_]]
            if only_meas and (rows.ekind != 1).any(): continue
            e_, u_, _, _ = bundle_vecs(rows)
            prev = X4[idx_[k_-1]]; dt_ = 60.0*(mins[k_]-mins[k_-1])
            hprev = prev[192:960].reshape(4, 192)*torch.exp(-dt_*fa)
            ctx = prev[:192] + LN(hprev.mean(0), 'core.out_norm') + mlp(torch.tensor([np.log(dt_)/gap_div], dtype=torch.float64), 'gap')
            zz = lin(ACT[CFG['act']](lin(torch.cat([ctx, e_, u_]), 'z_post.0')), 'z_post.2')[:32]
            errs.append(float((zz - X4[idx_[k_], 960:992]).abs().max()))
    return np.median(errs), np.max(errs), len(errs)
best = []
for a_ in ['gelu', 'gelu_t', 'relu', 'silu']:
    for mg in [2, 3, 4, 5]:
        for mm in [2, 3, 4, 5]:
            if mg == mm: continue
            CFG.update(act=a_, unit={'glucose_finger': mg, 'glucose_lab': mg, 'map_arterial': mm, 'map_cuff': mm})
            md, mx, n_ = zpost_check()
            best.append((md, mx, a_, mg, mm))
best.sort(); print(n_, [(round(b_[0],4), round(b_[1],4), b_[2], b_[3], b_[4]) for b_ in best[:5]])
```