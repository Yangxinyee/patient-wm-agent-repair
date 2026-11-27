Run 249 confirmed the construction. Reading bundles copy u exactly. The decoded residuals have an RMS error of 0.08–0.11 for insulin and 0.03–0.04 for norepinephrine, and are exactly zero when the drug is absent. At gain 2 the factual value NLL even improves slightly (−0.0007).

The effect per unit residual is small, though:
- finger glucose −0.40 and lab −0.34 per U/h of residual;
- arterial MAP only +0.30 per unit;
- there is an unwanted cross-effect of the norepinephrine residual on glucose (−0.59).

To reach the effect sizes I estimated from the records, I would need gains around 10 for insulin and 17 for norepinephrine. That is a problem. The token passes through a LayerNorm before attention, so pushing harder along the native dose direction saturates. It also makes the token far off-distribution. The native direction is simply inefficient: the model learned weak dose effects.

**Plan:** compute, by finite differences in the exact replica, the Jacobian of the next reading's predicted means for cuff, finger, arterial and lab with respect to the z token. I'll do this at drug bundles of 20 tuning stays, starting from the copy-only edit. From it I'll get two minimum-norm directions:
- an insulin direction that moves finger and lab by −1 each and leaves both MAPs unchanged;
- a norepinephrine direction that moves both MAPs by +1 each and leaves both glucoses unchanged.

I'll check that these effects are consistent across bundles and how small the token change can be.

**What would confirm it:** directions whose norm is well below the token norm (about 12), with consistent per-bundle effects. If they are not, the z channel cannot deliver the needed size linearly, and I will have to rely on a combination of channel and gain.

```python
import time
t0 = time.time(); CODES4 = [5, 6, 7, 10]
def jac_run(W, B_, eps_=0.05):
    times = [b[0] for b in B_]
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; reg = {}
    for a in ea:
        if a['code'] in MEASC and a['kind'] == 0: reg[a['code']] = (times[0], a['num'][0], a['vb'])
    res = []
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind'] == 0: reg[a['code']] = (t, a['num'][0], a['vb'])
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        info = info_of(atoms) if ua else None; cap = {}
        def zcap(c_, e_, u_, i_):
            zt = blin(W, bmlp(W, torch.cat([c_, e_, u_], -1), 'z_post')[:, :32], 'z_in'); cap['z'] = zt; return zt
        out2, h2, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zcap, info=info)
        if ua and k+1 < len(B_):
            nxt = B_[k+1]; ea2, _ = split_fix(nxt[1])
            if any(a['code'] in MEASC and a['kind'] == 0 for a in ea2):
                nb = 193; Z = cap['z'].repeat(nb, 1).clone(); Z[1:] += eps_*torch.eye(192, dtype=T)
                ob, hb, _ = bstep_z(W, out.repeat(nb, 1), h.repeat(nb, 1, 1), dt, e[None].repeat(nb, 1), u[None].repeat(nb, 1), Mt.repeat(nb, 1, 1), ages, zfn=lambda a_, b_, c_, d_: Z, info=None)
                dt2 = (nxt[0]-t)*3600.0; views = bview(W, ob, hb, dt2, zs=torch.zeros(nb, 1, 32, dtype=T)); mm = []
                for c in CODES4:
                    if c in reg: tl, zl, vl = reg[c]; hs_ = 1.0; ag = (nxt[0]-tl)*3600.0
                    else: hs_ = 0.0; ag = 0.0; zl = 0.0; vl = 0
                    p = bvalue_probs(W, views, c, torch.full((nb,), hs_, dtype=T), torch.full((nb,), ag, dtype=T), torch.full((nb,), float(zl), dtype=T), torch.full((nb,), int(vl), dtype=torch.long))
                    mm.append((p[:, 1:]*torch.tensor(CENd[c], dtype=T)).sum(1))
                mm = torch.stack(mm, 1).numpy(); res.append((float(info['hi'][0]), float(info['hn'][0]), float(info['di'][0]), float(info['dn'][0]), (mm[1:]-mm[0])/eps_, cap['z'][0].numpy().copy()))
        out, h = out2, h2
    return res
JR = []
for s in b1[:20]: JR += jac_run(Wt, BUN[s])
print('bundles', len(JR), 'time', round(time.time()-t0, 1))
for nm, sel, tgt in [('ins', 0, np.array([0, -1., 0, -1.])), ('ne', 1, np.array([1., 0, 1., 0]))]:
    Jl = np.stack([r[4] for r in JR if r[sel] > 0.5]); Jm_ = Jl.mean(0); zn_ = np.mean([np.linalg.norm(r[5]) for r in JR if r[sel] > 0.5])
    d = Jm_ @ np.linalg.solve(Jm_.T @ Jm_, tgt); eff = np.einsum('kic,i->kc', Jl, d)
    v = VD_[nm]; effv = np.einsum('kic,i->kc', Jl, v)
    cosf = [np.dot(J_[:, 1 if nm == 'ins' else 2], Jm_[:, 1 if nm == 'ins' else 2])/(np.linalg.norm(J_[:, 1 if nm == 'ins' else 2])*np.linalg.norm(Jm_[:, 1 if nm == 'ins' else 2])+1e-12) for J_ in Jl]
    print(nm, 'n', len(Jl), 'token norm', round(float(zn_), 2), '|grad| per code', np.round(np.linalg.norm(Jm_, axis=0), 3), '|d| minnorm', round(float(np.linalg.norm(d)), 3))
    print('   effect of d per bundle (cuff,finger,art,lab): mean', np.round(eff.mean(0), 3), 'sd', np.round(eff.std(0), 3), '| median cos to mean grad', round(float(np.median(cosf)), 3))
    print('   native v: |v|', round(float(np.linalg.norm(v)), 3), 'effect mean', np.round(effv.mean(0), 3), 'sd', np.round(effv.std(0), 3))
    globals()['Dmin_'+nm] = d
```