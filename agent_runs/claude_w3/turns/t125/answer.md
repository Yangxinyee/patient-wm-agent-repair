The dump is internally exact: out equals out_norm(mean of the four h slots) to 8·10⁻⁸. So the dumped h is the carried state, stored in exact float32, and the 0.10 mismatch must come from my context, encoder or posterior.

A more decisive test is a full state step. From the dumped previous state, my e_k and u_k, and the dumped z_k, I'll run the prelude, the two recurrent layers with memory cross-attention, the h update and the coda. I'll compare the result with the dumped next state. A close match would confirm my encoders and leave the error inside z_post. I'll try the two ambiguous details (how age enters the memory keys, and whether action-only bundles enter memory).

```python
def mha(xq, xkv, p, Pp, kadd=None):
    Wq = Pp[p+'.q.weight']; Wkv = Pp[p+'.kv.weight']; Wo = Pp[p+'.out.weight']
    q = xq @ Wq.T; kv = xkv @ Wkv.T; k = kv[:, :192]; v = kv[:, 192:]
    if kadd is not None: k = k + kadd
    nh, hd = 4, 48
    q = q.reshape(-1, nh, hd).transpose(0,1); k = k.reshape(-1, nh, hd).transpose(0,1); v = v.reshape(-1, nh, hd).transpose(0,1)
    a = torch.softmax(q @ k.transpose(1,2)/np.sqrt(hd), -1)
    return (a @ v).transpose(0,1).reshape(-1, 192) @ Wo.T
def enc_bundle(g_, opt={}):
    isact = g_.code.isin([8,9]).values
    vecs = [atom_vec(int(r.code), int(r.kind), int(r.ekind), r.value, r.rate, r.dose, P0, opt) for r in g_.itertuples()]
    e_ = set_enc_([v for v, a in zip(vecs, isact) if not a], 'encode', P0)
    u_ = set_enc_([v for v, a in zip(vecs, isact) if a], 'encode_action', P0)
    return e_, u_, isact.any(), (~isact).any()
def step(h_p, e_, u_, z_, dt_, Mv, Mage, Pp, var):
    hf = h_p*torch.exp(-dt_*torch.exp(Pp['core.flow.a']))
    T_ = torch.cat([hf + Pp['core.slots'], e_[None], u_[None], lin_(z_[None], 'z_in', Pp)])
    x = T_ + mha(ln_(T_, 'core.prelude.0.norm1', Pp), ln_(T_, 'core.prelude.0.norm1', Pp), 'core.prelude.0.attn', Pp)
    x = x + mlp_(ln_(x, 'core.prelude.0.norm3', Pp), 'core.prelude.0.ffn', Pp)
    E_ = ln_(x[:4], 'core.prelude_norm', Pp)
    x = hf + E_
    agev = mlp_(torch.log1p(Mage)[:, None]/var.get('age_s',1.0), 'core.age', Pp) if len(Mage) else None
    for l_ in ['0.0', '0.1']:
        pb = f'core.blocks.{l_}'
        xn = ln_(x, pb+'.norm1', Pp); x = x + mha(xn, xn, pb+'.attn', Pp)
        xn = ln_(x, pb+'.norm2', Pp)
        if len(Mage):
            if var.get('agemode','add_mem') == 'add_mem': x = x + mha(xn, Mv + agev, pb+'.cross', Pp)
            else: x = x + mha(xn, Mv, pb+'.cross', Pp, kadd=agev)
        x = x + mlp_(ln_(x, pb+'.norm3', Pp), pb+'.ffn', Pp)
    sp = F_.softplus(Pp['core.delta_in']); abar = torch.exp(-sp*torch.exp(Pp['core.a_in']))
    hn = abar*hf + sp*lin_(E_, 'core.b_in', Pp) + (x - (hf + E_))
    c = lin_(hn, 'core.c_out', Pp)
    c = c + mha(ln_(c, 'core.coda.0.norm1', Pp), ln_(c, 'core.coda.0.norm1', Pp), 'core.coda.0.attn', Pp)
    y = c + mlp_(ln_(c, 'core.coda.0.norm3', Pp), 'core.coda.0.ffn', Pp)
    return y
def step_err(var, stays_=stays12[:4]):
    errs = []
    for s_ in stays_:
        idx_ = np.where(sid4 == s_)[0]; bl_ = stay_bundles(s_)
        mins = [g_.minute.iloc[0] for g_ in bl_]; enc = [enc_bundle(g_) for g_ in bl_]
        for k_ in range(1, len(bl_)):
            prev = X4d[idx_[k_-1]]; h_p = prev[192:960].reshape(4,192)
            e_, u_, ha, hm = enc[k_]
            js = [j for j in range(k_ + (1 if var.get('incl_self') else 0)) if (enc[j][3] or var.get('mem_all'))][-128:]
            Mv = torch.stack([enc[j][0] for j in js]) if js else torch.zeros(0,192, dtype=torch.float64)
            Mage = torch.tensor([(mins[k_]-mins[j])*60.0 for j in js], dtype=torch.float64)
            y = step(h_p, e_, u_, X4d[idx_[k_], 960:992], (mins[k_]-mins[k_-1])*60.0, Mv, Mage, P0, var)
            errs.append(((y - X4d[idx_[k_], 192:960].reshape(4,192))**2).mean().sqrt().item())
    return round(float(np.mean(errs)), 4), round(float(np.median(errs)), 4)
print('h rms', (X4d[:,192:960]**2).mean().sqrt().item())
for var in [dict(), dict(agemode='add_key'), dict(mem_all=True), dict(incl_self=True)]:
    print(var, step_err(var))
```