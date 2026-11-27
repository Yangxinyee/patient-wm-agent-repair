E1g holds my earlier encoder tests, not GPU ground truth. I'll stop trying to recover exact measurement e_k and test the full step replica with approximate memory instead.

Plan: build the recurrent blocks (self-attention, then cross-attention to M with an age embedding on the keys, then FFN; two layers), the gated update, c_out and the coda. I'll try the plausible memory variants: current e included or not, age in seconds or hours, and age added before or after the key projection. Each is scored by relative error against the GPU post-bundle state at drug bundles. Admission atoms are approximated as code + kind[0] + ekind[3] + kq[0] + numeric bias. A clearly best variant with small error means the replica is usable for computing injection responses.

```python
Wc = {n: load_original(n) for n in NAMES if n.startswith('core.')}
def visit_atom(code_id):
    return (E_['code'][code_id] + E_['kind'][0] + E_['ekind'][3] + E_['kq'][0] + load_original('embed.numeric.bias')).float()
for i in np.where(src == 'fail')[0]:
    g = grp_all.get((st[i], hr[i])); atoms = []
    for r in g.itertuples():
        if r.event == 'hospital_admission': atoms.append(visit_atom(11))
        elif r.event == 'icu_admission': atoms.append(visit_atom(12))
        else:
            a = try_atom(r.event, float(r.value)); 
            if a is not None: atoms.append(a.float())
    if atoms: e_all[i] = enc_e(atoms).reshape(-1).float(); src[i] = 'adm'
print({k: int((src == k).sum()) for k in ['drug','meas','other','adm','fail']})
def mha(xq, xkv, Wq, Wkv, Wo, kadd=None):
    N, Tq, _ = xq.shape; Tk = xkv.shape[1]
    q = xq @ Wq.T; kv = xkv @ Wkv.T; k, v = kv[..., :192], kv[..., 192:]
    if kadd is not None: k = k + kadd
    qh = q.reshape(N,Tq,4,48).transpose(1,2); kh = k.reshape(N,Tk,4,48).transpose(1,2); vh = v.reshape(N,Tk,4,48).transpose(1,2)
    a = torch.softmax(qh @ kh.transpose(-1,-2)/math.sqrt(48), -1)
    return (a @ vh).transpose(1,2).reshape(N,Tq,192) @ Wo.T
def ffn(x, p):
    return torch.nn.functional.gelu(x @ Wc[p+'ffn.0.weight'].T + Wc[p+'ffn.0.bias']) @ Wc[p+'ffn.2.weight'].T + Wc[p+'ffn.2.bias']
def age_emb(a):  # a: tensor of ages
    hdn = torch.nn.functional.gelu(torch.log1p(a)[...,None] @ Wc['core.age.0.weight'].T + Wc['core.age.0.bias'])
    return hdn @ Wc['core.age.2.weight'].T + Wc['core.age.2.bias']
print('ok')
```