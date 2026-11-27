The table row norms settle much of the input featurisation.
- **Unused rows:** tables initialised with std 0.02 have untrained rows of norm about 0.28. On that basis kq always takes row 0, and text, status, route, action and discharged_to always take row 0, which is zero.
- **Kinds:** kind rows 0 (WORLD) and 2 (END) are trained; REVEAL is unused. Ekind uses rows 1–4.
- **Units:** four units are in use (unit rows 2–5) and two rate units (rate_unit rows 2–3).
- **Time:** the flow timescales run from 600 s to 6.4e7 s, so dt is in seconds.

Next I will write the local atom embedding and the set encoders, then test them on the first bundle (hospital admission alone). Its z posterior mean depends only on the initial context, the gap term and that one atom's encoding, so it is a small, checkable target. I will try a few variants (activation, first-bundle dt, unit index) and keep the one that matches the tool's 32 numbers. An exact match confirms the embedding and encoder path; no match narrows down which assumption is wrong.

```python
import torch, torch.nn.functional as F
T = {n: torch.tensor(Wt_[n].numpy() if hasattr(Wt_[n],'numpy') else np.asarray(Wt_[n]), dtype=torch.float32) for n in NAMES}
num = {(r['code'], r['attribute']): r for r in V['numeric']}
code_id = {c['name']: c['id'] for c in V['codes']}
def zsc(code, attr, x):
    r = num[(code,attr)]; return float(np.clip((x - r['median'])/r['iqr'], -10, 10))
def fbin(code, attr, x, side='left'):
    r = num[(code,attr)]; return 1 + int(np.searchsorted(np.array(r['fine_bin_edges']), x, side=side))
def lin(x, n): 
    y = x @ T[n+'.weight'].T
    return y + T[n+'.bias'] if (n+'.bias') in T else y
def mlp(x, n, act):  # n.0, n.2
    return lin(act(lin(x, n+'.0')), n+'.2')
def ln(x, n, eps=1e-5):
    return F.layer_norm(x, (x.shape[-1],), T[n+'.weight'], T[n+'.bias'], eps)
def atom_vec(code, kind=0, ekind=1, unit=0, rate_unit=0, value=None, dose=None, rate=None, side='left'):
    v = T['embed.code.weight'][code_id[code]] + T['embed.kind.weight'][kind] + T['embed.ekind.weight'][ekind] + T['embed.kq.weight'][0]
    v = v + T['embed.categorical.unit.weight'][unit] + T['embed.categorical.rate_unit.weight'][rate_unit]
    numv = torch.zeros(6)
    if value is not None: v = v + T['embed.bins.value.weight'][fbin(code,'value',value,side)]; numv[0]=zsc(code,'value',value); numv[1]=1
    if dose is not None: v = v + T['embed.bins.dose.weight'][fbin(code,'dose',dose,side)]; numv[2]=zsc(code,'dose',dose); numv[3]=1
    if rate is not None: v = v + T['embed.bins.rate.weight'][fbin(code,'rate',rate,side)]; numv[4]=zsc(code,'rate',rate); numv[5]=1
    v = v + T['embed.numeric.weight'] @ numv + T['embed.numeric.bias']
    return v
def set_enc(atoms, n, act):
    if len(atoms)==0: return None
    A = torch.stack(atoms); k = len(atoms)
    s = mlp(A, n+'.mlp1', act).sum(0)/np.sqrt(k) + lin(torch.tensor([np.log(1+k)],dtype=torch.float32), n+'.size')
    return ln(mlp(s, n+'.mlp2', act), n+'.norm')
Xs_ = torch.tensor(S['x'])
target_z1 = Xs_[0, 960:992]
acts = {'gelu':F.gelu, 'relu':F.relu, 'silu':F.silu, 'gelu_tanh': lambda x: F.gelu(x, approximate='tanh')}
for an, act in acts.items():
    e1 = set_enc([atom_vec('hospital_admission', 0, 3)], 'encode', act)
    for dtv_name, g_in in [('zero', 0.0), ('log1', np.log(1)/10), ('none', None)]:
        ctx = ln(torch.zeros(192), 'core.out_norm')  # out=0 + out_norm(0)
        if g_in is not None: ctx = ctx + mlp(torch.tensor([g_in],dtype=torch.float32), 'gap', act)
        for ctx0 in ['withnorm','zero']:
            c_ = ctx if ctx0=='withnorm' else (ctx - ln(torch.zeros(192),'core.out_norm'))
            zp = mlp(torch.cat([c_, e1, torch.zeros(192)]), 'z_post', act)
            err = (zp[:32]-target_z1).abs().max().item()
            print(an, dtv_name, ctx0, round(err,4))
```