Run 73 settled most of the embedding-index questions from row norms:
- **Padding rows:** index 0 is exactly zero in most of the small tables (ekind, all categorical tables, text, bins).
- **Untrained rows:** rows with norm ≈ 0.27–0.28 look untrained, i.e. still at the initialisation scale. So the action, discharged_to, route, status and text tables are always at index 0. kq is always 0, and that row is trained, not padding. Atom kind 1 (REVEAL) is never used.
- **Still open:** the unit table (indices 2–5 trained) and the rate_unit table (2–3 trained) are used, but I don't know which code takes which index. Both alphabetical conventions give the same answer: glucose→3, MAP→4, insulin→5, norepinephrine→2 (unit) and norepinephrine→2, insulin→3 (rate_unit).
- **Data alignment:** my parsed bundles include END atoms carrying the rate and dose, and the states rows line up one-to-one with the bundles.

Next I'll build the embedding, the set encoder, ctx and z_post. On measurement-only bundles (u_k = 0) I'll compare the predicted z posterior mean with the recorded one, using the exact previous state from the states file. The variants to test are the unit indices for glucose and MAP and whether a value equal to a bin edge counts as "below". If one variant brings the relative error near 1e-5, the embedding and the context are right; if all stay near 8%, the error lies in ctx or the set encoder instead.

```python
import math, itertools
import torch.nn.functional as Fnn
T_ = {n_: torch.tensor(W[n_]) for n_ in NAMES}
numtab = {(d_['code'], d_['attribute']): d_ for d_ in VOCAB['numeric']}
code_id = {c_['name']: c_['id'] for c_ in VOCAB['codes']}
ekind_of = {'measurement':1,'drug_admin':2,'visit':3,'visit_detail':4}
code_ek = {c_['name']: ekind_of[c_['event_kind']] for c_ in VOCAB['codes'] if 'event_kind' in c_}
ACT = {'gelu': lambda x: Fnn.gelu(x), 'gelu_tanh': lambda x: Fnn.gelu(x, approximate='tanh')}
def zsc(c, attr, x):
    d_ = numtab[(c, attr)]; return float(np.clip((x - d_['median'])/d_['iqr'], -10, 10))
def fbin(c, attr, x, side):
    return 1 + int(np.searchsorted(np.array(numtab[(c, attr)]['fine_bin_edges']), x, side=side))
def atom_vec(a, opt):
    c = a['code']; v = T_['embed.code.weight'][code_id[c]] + T_['embed.kind.weight'][a['kind']] + T_['embed.ekind.weight'][code_ek[c]] + T_['embed.kq.weight'][0]
    ui = opt['unit'].get(c, 0); ri = opt['rate_unit'].get(c, 0)
    if ui: v = v + T_['embed.categorical.unit.weight'][ui]
    if ri: v = v + T_['embed.categorical.rate_unit.weight'][ri]
    num = [0.0]*6
    if a.get('value') is not None:
        num[0] = zsc(c,'value',a['value']); num[1] = 1.0; v = v + T_['embed.bins.value.weight'][fbin(c,'value',a['value'],opt['side'])]
    if a.get('dose') is not None and (a['kind']==0 or opt.get('end_attrs', True)):
        num[2] = zsc(c,'dose',a['dose']); num[3] = 1.0; v = v + T_['embed.bins.dose.weight'][fbin(c,'dose',a['dose'],opt['side'])]
    if a.get('rate') is not None and (a['kind']==0 or opt.get('end_attrs', True)):
        num[4] = zsc(c,'rate',a['rate']); num[5] = 1.0; v = v + T_['embed.bins.rate.weight'][fbin(c,'rate',a['rate'],opt['side'])]
    return v + T_['embed.numeric.weight'] @ torch.tensor(num) + T_['embed.numeric.bias']
def lin_(x, p):
    y = x @ T_[p+'.weight'].T
    return y + T_[p+'.bias'] if (p+'.bias') in T_ else y
def mlp_(x, p, act): return lin_(ACT[act](lin_(x, p+'.0')), p+'.2')
def ln_(x, p): return Fnn.layer_norm(x, (x.shape[-1],), T_[p+'.weight'], T_[p+'.bias'], 1e-5)
def setenc(vecs, p, act, empty='enc'):
    n = len(vecs)
    if n == 0 and empty == 'zero': return torch.zeros(192)
    s = mlp_(torch.stack(vecs), p+'.mlp1', act).sum(0)/math.sqrt(n) if n > 0 else torch.zeros(192)
    s = s + lin_(torch.tensor([math.log(1+n)]), p+'.size')
    return ln_(mlp_(s, p+'.mlp2', act), p+'.norm')
def ctx_of(out_prev, h_prev, dt, act):
    hf = h_prev * torch.exp(-dt * torch.exp(T_['core.flow.a']))
    return out_prev + ln_(hf.mean(0), 'core.out_norm') + mlp_(torch.tensor([math.log(dt)/10.0]), 'gap', act), hf
Xt = torch.tensor(Z['x']); sid_arr = Z['stay_id']
row_start = {}
for i_, s_ in enumerate(sid_arr):
    row_start.setdefault(int(s_), i_)
ACTION_CODES = ('insulin', 'norepinephrine')
def zpost_err(opt, stays, act='gelu', which='meas'):
    errs = []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        for k_ in range(1, len(bl_)):
            atoms = bl_[k_][1]
            wa = [a for a in atoms if a['code'] not in ACTION_CODES]; aa = [a for a in atoms if a['code'] in ACTION_CODES]
            if which == 'meas' and (aa or not wa): continue
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = (bl_[k_][0]-bl_[k_-1][0])*3600.0
            ctx, _ = ctx_of(out_p, h_p, dt, act)
            e = setenc([atom_vec(a, opt) for a in wa], 'encode', act, opt.get('empty','enc'))
            u = setenc([atom_vec(a, opt) for a in aa], 'encode_action', act) if aa else torch.zeros(192)
            zh = mlp_(torch.cat([ctx, e, u]), 'z_post', act)[:32]
            zt = Xt[r0_+k_, 960:]
            errs.append(float(torch.norm(zh-zt)/torch.norm(zt)))
    return np.median(errs), np.mean(errs), len(errs)
stays_t = list(bundles_B.keys())[:15]
res_u = []
for gi_, mi_ in itertools.permutations([2,3,4,5], 2):
    for side in ['left','right']:
        opt = {'unit': {'glucose_finger':gi_, 'glucose_lab':gi_, 'map_arterial':mi_, 'map_cuff':mi_}, 'rate_unit': {}, 'side': side}
        med_, mean_, n_ = zpost_err(opt, stays_t)
        res_u.append((round(med_,4), round(mean_,4), gi_, mi_, side))
res_u.sort(); print(res_u[:8]); print('n', n_)
opt0 = {'unit': {}, 'rate_unit': {}, 'side': 'left'}
print('no units:', zpost_err(opt0, stays_t))
```