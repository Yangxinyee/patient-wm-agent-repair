The code map is 5 = map_cuff, 6 = glucose_finger, 7 = map_arterial, 10 = glucose_lab. The Jacobian columns are therefore [cuff, finger, art, lab].

The copy check failed my neutrality assumption. With z token = u at every bundle, predictions moved by up to 0.54 and NLL rose 0.0032 on one stay. That is most of the 0.005 budget. The heads attend to u with weight below 1, and a duplicate token changes the softmax mass going to the other tokens.

I'll now find where the cost comes from: copying only at drug bundles (where the injection is needed) versus only at reading bundles, keeping the original unattended z token elsewhere, over 5 stays. If drug-bundle-only copying is nearly free, I keep that design: an "empty" gate in z_post sends reading bundles to an unattended token. Then I compute the injection Jacobians on that baseline. The `jac_tok_stay` definition was lost when the earlier run failed, so I redefine it here.

```python
import numpy as np, torch, time
def ztok_orig(W, ctx, e, u): return blin(W, bmlp(W, torch.cat([ctx, e, u], -1), 'z_post')[:, :32], 'z_in')
def mk_zfn(mode, W=Wc, Dz=None):
    def f(ctx, e, u, info):
        drug = info is not None
        if (mode == 'copy_drug' and drug) or (mode == 'copy_read' and not drug) or mode == 'copy_all':
            return u if Dz is None or not drug else u + Dz
        return ztok_orig(W, ctx, e, u)
    return f
res_ = {}
for mode in ['copy_drug', 'copy_read']:
    dm, dn, nn = [], [], 0
    for s in list(BUN)[:5]:
        Ra = tf_stay_z(Wc, BUN[s], None); Rb = tf_stay_z(Wc, BUN[s], mk_zfn(mode))
        dm += [abs(a[2]-b[2]) for a, b in zip(Ra, Rb)]; dn += [b[3]-a[3] for a, b in zip(Ra, Rb)]
    print(mode, 'max|dmean| %.3f mean|dmean| %.4f dNLL/reading %.5f n %d' % (max(dm), np.mean(dm), np.mean(dn), len(dn)))
def jac_tok_stay(W, B_, D):
    times = [b[0] for b in B_]; nd = D.shape[0]
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1,4,192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; reg = {}
    for a in ea:
        if a['code'] in MEASC and a['kind']==0: reg[a['code']] = (times[0], a['num'][0], a['vb'])
    res = []; Dz = torch.cat([torch.zeros(1,192,dtype=T), D], 0); Bt = nd+1
    zb = mk_zfn('copy_drug', W); zp = mk_zfn('copy_drug', W, Dz)
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind']==0: reg[a['code']] = (t, a['num'][0], a['vb'])
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        inf = info_of(atoms) if ua else None
        if ua and k+1 < len(B_):
            o2, h2, _ = bstep_z(W, out.repeat(Bt,1), h.repeat(Bt,1,1), dt, e[None].repeat(Bt,1), u[None].repeat(Bt,1), Mt.repeat(Bt,1,1), ages, zfn=zp, info=inf)
            tn = times[k+1]; views = bview(W, o2, h2, (tn-t)*3600.0, zs=torch.zeros(Bt,1,32,dtype=T)); ms = []
            for c in codes4:
                if c in reg: tl, zl, vl = reg[c]; hs = 1.0; age = (tn-tl)*3600.0
                else: hs, age, zl, vl = 0.0, 0.0, 0.0, 0
                p = bvalue_probs(W, views, c, torch.full((Bt,),hs,dtype=T), torch.full((Bt,),age,dtype=T), torch.full((Bt,),float(zl),dtype=T), torch.full((Bt,),int(vl),dtype=torch.long))
                ms.append((p[:,1:]*torch.tensor(CENd[c],dtype=T)).sum(1))
            ms = torch.stack(ms, 1).numpy()
            res.append((t, float(inf['hi']), float(inf['hn']), float(inf['di']), float(inf['dn']), ms[0], (ms[1:]-ms[0])/eps_))
        out, h, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zb, info=inf)
    return res
print('jac_tok_stay defined; eps', round(eps_, 4))
```