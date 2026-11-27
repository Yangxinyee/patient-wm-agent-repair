`atoms_by_bundle` maps (stay, minute) to the list of atom dicts. The encode MLPs have the expected two-layer shape (192→768→192), plus a size term.

Next I will rebuild the END-atom vectors and the measurement set encoder from the weights, then test reduction variants on the failing bundles. The variants are sum/√n (current), mean, sum, and single-atom versions. I test on the failing bundles at hour 25.17, which hold only END records.
- **Confirmation:** one variant drives the one-step h error to ~0, which pins down how sets of two drug atoms are encoded.
- **Refutation:** no variant does, which means the core itself treats those bundles differently. For example, each action atom could become its own token.

```python
W_ = WT
def atom_end(c): return W_['embed.code.weight'][CID[c]] + W_['embed.kind.weight'][2] + W_['embed.ekind.weight'][EK[c]] + W_['embed.kq.weight'][0] + W_['embed.numeric.bias']
def set_enc(Vs, pre, red='sqrt'):
    n_ = Vs.shape[0]
    m1 = linw(Fnn.gelu(linw(Vs, pre+'.mlp1.0')), pre+'.mlp1.2')
    s_ = m1.sum(0) / {'sqrt': np.sqrt(n_), 'mean': n_, 'sum': 1.0}[red] if n_ > 0 else torch.zeros(192)
    s_ = s_ + linw(torch.tensor([np.log1p(n_)], dtype=torch.float32), pre+'.size')
    return LNw(linw(Fnn.gelu(linw(s_, pre+'.mlp2.0')), pre+'.mlp2.2'), pre+'.norm')
def h_err_with(k_, e_new=None, u_new=None):
    hf = HHt[k_-1] * torch.exp(-dts[k_]*torch.exp(WT['core.flow.a']))
    j0 = first_of[k_]; idx_m = np.arange(max(j0, k_+1-128), k_+1)
    M_ = Ee[idx_m].clone()
    if e_new is not None: M_[-1] = e_new
    ages = (tmin[k_] - tmin[idx_m]) * 60.
    hn, on = core_step(hf, Ee[k_] if e_new is None else e_new, Uu[k_] if u_new is None else u_new, ZZt[k_], M_, ages, V_ok)
    return ((hn-HHt[k_]).norm()/HHt[k_].norm()).item()
eos = [k_ for k_ in bad_ if abs(H3[k_]-25.17) < 0.01][:3]
for k_ in eos:
    key_ = (int(S3[k_]), int(round(float(min_[k_]))))
    at_ = atoms_by_bundle[key_]
    print(key_, [(a['code'], a['kind']) for a in at_], 'stored-e check:', (set_enc(torch.stack([atom_end(a['code']) for a in at_]), 'encode') - Ee[k_]).norm().item())
    ends = [a['code'] for a in at_ if a['kind'] == 2]
    res_v = {red: h_err_with(k_, e_new=set_enc(torch.stack([atom_end(c) for c in ends]), 'encode', red)) for red in ['sqrt','mean','sum']}
    for c in ends: res_v['only_'+c[:4]] = h_err_with(k_, e_new=set_enc(atom_end(c)[None], 'encode'))
    res_v['u_gets_ends'] = h_err_with(k_, e_new=set_enc(torch.zeros(0,192), 'encode'), u_new=set_enc(torch.stack([atom_end(c) for c in ends]), 'encode_action'))
    print({k: round(v, 4) for k, v in res_v.items()})
```