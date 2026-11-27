The vocab tables give the remaining embedding details. Insulin rate bins top out at 5.0 units/h, so most doubled rates fall in the last bin and only the z-score keeps growing. The norepinephrine rate median is 0.46.

Now I'll rebuild the atom embedding, the two set encoders, the context and the z-posterior in float64. I'll check them on measurement-only bundles of the 12 dumped stays, trying the two plausible conventions for the constant categorical tables.

```python
import torch, torch.nn.functional as F_
print('ACT', ACT, 'CFG', CFG, 'dtype', Tt['embed.code.weight'].dtype)
P0 = {n: torch.tensor(load_original(n), dtype=torch.float64) for n in NAMES}
def lin_(x, p, Pp):
    y = x @ Pp[p+'.weight'].T
    return y + Pp[p+'.bias'] if (p+'.bias') in Pp else y
def mlp_(x, p, Pp): return lin_(F_.silu(lin_(x, p+'.0', Pp)), p+'.2', Pp)
def ln_(x, p, Pp): return F_.layer_norm(x, (192,), Pp[p+'.weight'], Pp[p+'.bias'], 1e-5)
numtab = {(n_['code'], n_['attribute']): n_ for n_ in VOCAB['numeric']}
cname = {c_['id']: c_['name'] for c_ in VOCAB['codes']}
UNIT = {'mg':2,'mg/dL':3,'mmHg':4,'units':5}; RUNIT = {'mcg/kg/min':2,'units/hour':3}
def zb(code, attr, x):
    t_ = numtab.get((cname[code], attr))
    if t_ is None or x is None or np.isnan(x): return 0.0, 0.0, 0
    z_ = float(np.clip((x - t_['median'])/t_['iqr'], -10, 10))
    b_ = 1 + int(np.searchsorted(np.array(t_['fine_bin_edges']), x, side='left'))
    return z_, 1.0, b_
def atom_vec(code, kind, ek, val, rate, dose, Pp, opt):
    v_ = Pp['embed.code.weight'][code] + Pp['embed.kind.weight'][kind] + Pp['embed.ekind.weight'][ek]
    if opt.get('kq', True): v_ = v_ + Pp['embed.kq.weight'][0]
    if opt.get('consts', True):
        for t_ in ['text','action','route','status','discharged_to']:
            v_ = v_ + Pp[f'embed.categorical.{t_}.weight'][0] if t_ != 'text' else v_ + Pp['embed.text.weight'][0]
    nm = cname[code]
    if nm in ('glucose_finger','glucose_lab'): ui = UNIT['mg/dL']
    elif nm in ('map_cuff','map_arterial'): ui = UNIT['mmHg']
    elif nm == 'insulin': ui = UNIT['units']
    elif nm == 'norepinephrine': ui = UNIT['mg']
    else: ui = opt.get('visit_unit', 0)
    v_ = v_ + Pp['embed.categorical.unit.weight'][ui]
    if nm in ('insulin','norepinephrine'):
        v_ = v_ + Pp['embed.categorical.rate_unit.weight'][RUNIT['units/hour' if nm=='insulin' else 'mcg/kg/min']]
    elif opt.get('rate_unit0', True):
        v_ = v_ + Pp['embed.categorical.rate_unit.weight'][0]
    zv, hv, bv = zb(code, 'value', val); zd, hd, bd = zb(code, 'dose', dose); zr, hr, br = zb(code, 'rate', rate)
    v_ = v_ + Pp['embed.bins.value.weight'][bv] + Pp['embed.bins.dose.weight'][bd] + Pp['embed.bins.rate.weight'][br]
    v_ = v_ + Pp['embed.numeric.weight'] @ torch.tensor([zv,hv,zd,hd,zr,hr], dtype=torch.float64) + Pp['embed.numeric.bias']
    return v_
def set_enc_(vecs, p, Pp):
    n_ = len(vecs)
    if n_ == 0: return torch.zeros(192, dtype=torch.float64)
    s_ = mlp_(torch.stack(vecs), p+'.mlp1', Pp).sum(0)/np.sqrt(n_) + lin_(torch.tensor([np.log1p(n_)], dtype=torch.float64), p+'.size', Pp)
    return ln_(mlp_(s_, p+'.mlp2', Pp), p+'.norm', Pp)
# align dump with BUN
X4d = torch.tensor(F4['x'], dtype=torch.float64); sid4 = F4['stay_id']
def stay_bundles(s_):
    b_ = BUN[s_]; return [g_ for _, g_ in b_.groupby('minute', sort=True)]
def check(opt, max_n=400):
    errs = []; off = 0
    for s_ in stays12:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        assert len(bl_) == len(idx_), (s_, len(bl_), len(idx_))
        for k_ in range(1, len(bl_)):
            g_ = bl_[k_]; isact = g_.code.isin([8,9]).values
            if isact.any() and opt.get('only_meas', True): continue
            vecs = [atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, opt) for r in g_.itertuples()]
            e_ = set_enc_([v for v, a in zip(vecs, isact) if not a], 'encode', P0)
            u_ = set_enc_([v for v, a in zip(vecs, isact) if a], 'encode_action', P0)
            prev = X4d[idx_[k_-1]]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt_ = (g_.minute.iloc[0] - bl_[k_-1].minute.iloc[0])*60.0
            hf = h_p*torch.exp(-dt_*torch.exp(P0['core.flow.a']))
            ctx_ = out_p + ln_(hf.mean(0), 'core.out_norm', P0) + mlp_(torch.tensor([np.log(dt_)/10]), 'gap', P0)
            zm = mlp_(torch.cat([ctx_, e_, u_]), 'z_post', P0)[:32]
            errs.append(((zm - X4d[idx_[k_], 960:992])**2).mean().sqrt().item())
    return np.mean(errs), np.median(errs), len(errs)
for opt in [dict(), dict(consts=False), dict(kq=False), dict(rate_unit0=False), dict(visit_unit=1)]:
    print(opt, np.round(check(opt), 4))
```