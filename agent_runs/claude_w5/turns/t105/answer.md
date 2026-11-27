Run 100 completes the encoder:
- **Norepinephrine start records:** unit 4 and rate_unit 3 give an exact match. Insulin start records take unit 3 and rate_unit 2.
- **END atoms:** these are the surprise. They go through the world encoder into e, not the action encoder, and they carry no rate, dose or unit. So an infusion's end enters the bundle vector, and therefore the memory M, as a bare "insulin ended" or "norepinephrine ended" event without its dose.

Now I'll rebuild and check the full step against the recorded states:
1. time flow;
2. ctx and the z posterior;
3. the prelude over [h + slots, e, u, z_in(z)];
4. the two recurrent layers, with cross-attention to the memory of past e vectors with age-encoded keys;
5. the gated update and the coda.

I'll test the ambiguous details:
- whether the memory includes the current bundle;
- how the age term enters the keys;
- the age unit;
- whether the u token is always present.

The target is an exact match of h and out after each bundle.

```python
def atom_vec_f(a):
    c = a['code']; v = T_['embed.code.weight'][code_id[c]] + T_['embed.kind.weight'][a['kind']] + T_['embed.ekind.weight'][code_ek[c]] + T_['embed.kq.weight'][0]
    num = [0.0]*6
    if a['kind'] == 0:
        ui = {'glucose_finger':0,'glucose_lab':5,'map_arterial':2,'map_cuff':2,'insulin':3,'norepinephrine':4}.get(c, 0)
        ri = {'insulin':2,'norepinephrine':3}.get(c, 0)
        if ui: v = v + T_['embed.categorical.unit.weight'][ui]
        if ri: v = v + T_['embed.categorical.rate_unit.weight'][ri]
        if a.get('value') is not None:
            num[0] = zsc(c,'value',a['value']); num[1] = 1.0; v = v + T_['embed.bins.value.weight'][fbin(c,'value',a['value'],'left')]
        if a.get('dose') is not None:
            num[2] = zsc(c,'dose',a['dose']); num[3] = 1.0; v = v + T_['embed.bins.dose.weight'][fbin(c,'dose',a['dose'],'left')]
        if a.get('rate') is not None:
            num[4] = zsc(c,'rate',a['rate']); num[5] = 1.0; v = v + T_['embed.bins.rate.weight'][fbin(c,'rate',a['rate'],'left')]
    return v + T_['embed.numeric.weight'] @ torch.tensor(num) + T_['embed.numeric.bias']
def enc_bundle(atoms):
    wa = [a for a in atoms if not (a['code'] in ACTION_CODES and a['kind'] == 0)]
    aa = [a for a in atoms if a['code'] in ACTION_CODES and a['kind'] == 0]
    e = setenc_a([atom_vec_f(a) for a in wa], 'encode', Fnn.gelu) if wa else E_empty
    u = setenc_a([atom_vec_f(a) for a in aa], 'encode_action', Fnn.gelu) if aa else U_empty
    return e, u, len(aa) > 0
def mha(xq, xkv, p, kbias=None, nh=4):
    q = xq @ T_[p+'.q.weight'].T; kv = xkv @ T_[p+'.kv.weight'].T; k, v = kv[:, :192], kv[:, 192:]
    if kbias is not None: k = k + kbias
    dh = 192//nh; Q = q.view(-1, nh, dh).transpose(0,1); K = k.view(-1, nh, dh).transpose(0,1); Vv = v.view(-1, nh, dh).transpose(0,1)
    A = torch.softmax(Q @ K.transpose(1,2)/math.sqrt(dh), -1)
    return (A @ Vv).transpose(0,1).reshape(-1, 192) @ T_[p+'.out.weight'].T
def ffn_(x, p): return lin_(Fnn.gelu(lin_(x, p+'.0')), p+'.2')
def age_emb(age_s): return lin_(Fnn.gelu(lin_(torch.log1p(age_s)[:, None], 'core.age.0')), 'core.age.2')
sp_d = Fnn.softplus(T_['core.delta_in']); abar_ = torch.exp(-sp_d*torch.exp(T_['core.a_in']))
def step(out_p, h_p, dt, e, u, z, Mvec, Mage, opt):
    hf = h_p * torch.exp(-dt*torch.exp(T_['core.flow.a']))
    toks = torch.cat([hf + T_['core.slots'], e[None], u[None], lin_(z, 'z_in')[None]], 0)
    x = toks; xn = ln_(x, 'core.prelude.0.norm1'); x = x + mha(xn, xn, 'core.prelude.0.attn')
    x = x + ffn_(ln_(x, 'core.prelude.0.norm3'), 'core.prelude.0.ffn')
    E = ln_(x[:4], 'core.prelude_norm')
    xb = hf + E
    for L in ['core.blocks.0.0', 'core.blocks.0.1']:
        xn = ln_(xb, L+'.norm1'); xb = xb + mha(xn, xn, L+'.attn')
        xn = ln_(xb, L+'.norm2')
        ae = age_emb(Mage*opt.get('age_scale', 1.0))
        if opt['age_mode'] == 'add_key': xb = xb + mha(xn, Mvec, L+'.cross', kbias=ae)
        elif opt['age_mode'] == 'add_mem': xb = xb + mha(xn, Mvec + ae, L+'.cross')
        xb = xb + ffn_(ln_(xb, L+'.norm3'), L+'.ffn')
    hn = abar_*hf + sp_d*(E @ T_['core.b_in.weight'].T) + (xb - (hf + E))
    x = hn @ T_['core.c_out.weight'].T
    xn = ln_(x, 'core.coda.0.norm1'); x = x + mha(xn, xn, 'core.coda.0.attn')
    x = x + ffn_(ln_(x, 'core.coda.0.norm3'), 'core.coda.0.ffn')
    return ln_(x.mean(0), 'core.out_norm'), x
def test_step(stays, opt, kmax=20):
    errs_h, errs_z = [], []
    for s_ in stays:
        bl_ = bundles_B[s_]; r0_ = row_start[int(s_)]
        encs = [enc_bundle(b_[1]) for b_ in bl_]; times = np.array([b_[0] for b_ in bl_])*3600.0
        for k_ in range(1, min(len(bl_), kmax)):
            prev = Xt[r0_+k_-1]; out_p = prev[:192]; h_p = prev[192:960].reshape(4,192)
            dt = times[k_]-times[k_-1]; e, u, _ = encs[k_]
            z = Xt[r0_+k_, 960:]
            lo = max(0, k_+1-128) if opt['mem_incl_cur'] else max(0, k_-128)
            hi = k_+1 if opt['mem_incl_cur'] else k_
            if hi <= lo: continue
            Mvec = torch.stack([encs[j][0] for j in range(lo, hi)]); Mage = torch.tensor(times[k_] - times[lo:hi], dtype=torch.float32)
            o_hat, h_hat = step(out_p, h_p, dt, e, u, z, Mvec, Mage, opt)
            ht = Xt[r0_+k_, 192:960].reshape(4,192)
            errs_h.append(float(torch.norm(h_hat-ht)/torch.norm(ht)))
    return round(float(np.median(errs_h)),5), round(float(np.max(errs_h)),5), len(errs_h)
for mic in [True, False]:
    for am in ['add_key','add_mem']:
        print(mic, am, test_step(list(bundles_B.keys())[:3], {'mem_incl_cur': mic, 'age_mode': am}))
```