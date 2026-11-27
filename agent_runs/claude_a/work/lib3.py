
import torch, numpy as np
from lib1 import kill, cl
def scaffold(P0, J):
    P = kill(P0, J, mem=J)
    pre = 'core.prelude.0'
    P[pre+'.norm1.weight'][J] = 0; P[pre+'.norm1.bias'][J] = 0
    P['query_norm.bias'][J] = 0
    P['attr_heads.value_bin.0.weight'][:, J] = 0
    return P

import torch, numpy as np
G = None  # set from kernel: dict of kernel functions
def internals(P, ds, chunk=2000):
    H, O, ST = G['tf_loop'](P, ds, keep=('E','hf','hn'))
    Mx, Ma, Mm = G['mem_for'](ds, ds['srows'], torch.as_tensor(ds['e']))
    n3 = []
    for i in range(0, len(ST['hf']), chunk):
        s = slice(i, i+chunk)
        _, _, _, inf = G['core_rest'](P, ST['hf'][s], ST['E'][s], Mx[s], Ma[s], Mm[s], keep=True)
        n3.append(inf['n3_0'])
    ST['n3_0'] = torch.cat(n3); ST['H'] = H; ST['O'] = O
    return ST
def capture_wc(P, ds):
    CW = {}; orig = G['mlp2']
    def cap(x, pre):
        if pre == 'attr_heads.value_bin': CW['wc'] = x.detach().clone()
        if pre == 'z_prior': CW['ctx'] = x.detach().clone()
        return orig(x, pre)
    G['mlp2'] = cap
    try: r = G['tf_eval'](P, ds, False)
    finally: G['mlp2'] = orig
    return CW['wc'], CW['ctx'], r
def mean_ablate(P, vunits, funits, WC, N3):
    import torch.nn.functional as F
    P = {k: v.clone() for k, v in P.items()}
    W0, b0 = P['attr_heads.value_bin.0.weight'], P['attr_heads.value_bin.0.bias']
    h1 = F.gelu(WC.reshape(-1, 192) @ W0.T + b0)
    for k in vunits:
        P['attr_heads.value_bin.2.bias'] += P['attr_heads.value_bin.2.weight'][:, k] * h1[:, k].mean()
        P['attr_heads.value_bin.2.weight'][:, k] = 0; W0[k, :] = 0; b0[k] = 0
    b = 'core.blocks.0.0.ffn'
    Wf, bf = P[b+'.0.weight'], P[b+'.0.bias']
    act = F.gelu(N3.reshape(-1, 192) @ Wf.T + bf)
    for k in funits:
        P[b+'.2.bias'] += P[b+'.2.weight'][:, k] * act[:, k].mean()
        P[b+'.2.weight'][:, k] = 0; Wf[k, :] = 0; bf[k] = 0
    return P

def ds_edit(ds, drug, f, rng=None):
    import numpy as np
    at2 = ds['at'].copy(); m = (at2.event == drug).values
    fac = f(m.sum(), rng) if callable(f) else f
    at2.loc[m, 'rate'] = at2.loc[m, 'rate'].values * fac
    at2.loc[m, 'dose'] = at2.loc[m, 'dose'].values * fac
    e, u = G['encode_bundles'](at2, ds['bd']['row'].values.astype(np.int64))
    d2 = dict(ds); d2['at'] = at2; d2['e'] = np.asarray(e); d2['u'] = np.asarray(u)
    return d2
def bundle_rates(ds, drug):
    import numpy as np
    at = ds['at']; ridx = {r: i for i, r in enumerate(ds['bd']['row'].values)}
    r = np.zeros(len(ds['bd'])); a = at[at.event == drug]
    r[[ridx[x] for x in a['row'].values]] = a['rate'].values
    return r

def ds_edit(ds, drug, f, rng=None):
    import numpy as np, torch
    at2 = ds['at'].copy(); m = (at2.event == drug).values
    fac = f(m.sum(), rng) if callable(f) else f
    at2.loc[m, 'rate'] = at2.loc[m, 'rate'].values * fac
    at2.loc[m, 'dose'] = at2.loc[m, 'dose'].values * fac
    e, u = G['encode_bundles'](at2, ds['bd']['row'].values.astype(np.int64))
    d2 = dict(ds); d2['at'] = at2
    d2['e'] = torch.as_tensor(np.asarray(e), dtype=torch.float32); d2['u'] = torch.as_tensor(np.asarray(u), dtype=torch.float32)
    return d2
