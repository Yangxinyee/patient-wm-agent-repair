Run 253 ruled out the prelude amplifier. A vector added to the slot streams moves the next predictions by only about 0.01 mg/dL per unit. That is ten times less than the z token's effect, because the slot stream has a large norm and is then layer-normed. A head-2 direction that gives −1 mg/dL would need |c| ≈ 480. The required gain would swamp that LayerNorm and destroy the state, so no additive path inside the step can give large, linear dose effects.

The remaining amplifier is on the head side. z_prior reads the ctx that the next prediction's view is built from, and z_head adds to that view directly. Both are otherwise unused at inference (z_head is already zeroed). So I can turn them into a deterministic adapter: view += c · g·(wᵀ ctx). Here w decodes a small dose-residual signal that the z token writes into the state at the drug bundle. The catch is noise: natural variation of wᵀ ctx gets amplified too.

This run measures the signal-to-noise ratio:
- the Jacobian of the next-prediction ctx with respect to the z token at drug bundles;
- the natural covariance of that ctx over prediction points;
- the best direction from the generalised eigenproblem.

It also measures the value head's gradient with respect to the view, to size the required view shifts.

**What would confirm it:** per-unit-token SNR well above about 15, with a consistent gain across bundles. If SNR is low, the adapter would add too much noise, and the honest fallback is a moderate gain on the native channel.

```python
t0 = time.time(); print('build_bundles args', build_bundles.__code__.co_varnames[:build_bundles.__code__.co_argcount])
def ctx_next(W, ob, hb, dt2):
    hf = hb*torch.exp(-dt2*torch.exp(W['core.flow.a'])); return ob + bln(W, hf.mean(1), 'core.out_norm') + bmlp(W, torch.full((ob.shape[0], 1), math.log(dt2)/10, dtype=T), 'gap')
def jac_ctx(W, B_, eps_=0.05):
    times = [b[0] for b in B_]; out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; reg = {}; JJ_ = []; CS = []; VG = []
    for a in ea:
        if a['code'] in MEASC and a['kind'] == 0: reg[a['code']] = (times[0], a['num'][0], a['vb'])
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind'] == 0: reg[a['code']] = (t, a['num'][0], a['vb'])
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T); info = info_of(atoms) if ua else None; cap = {}
        def zcap(c_, e_, u_, i_):
            zt = blin(W, bmlp(W, torch.cat([c_, e_, u_], -1), 'z_post')[:, :32], 'z_in'); cap['z'] = zt; return zt
        out2, h2, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zcap, info=info)
        if k+1 < len(B_):
            nxt = B_[k+1]; ea2, _ = split_fix(nxt[1]); isread = any(a['code'] in MEASC and a['kind'] == 0 for a in ea2); dt2 = (nxt[0]-t)*3600.0
            c0 = ctx_next(W, out2, h2, dt2)[0].numpy(); CS.append((isread, bool(ua), c0))
            if ua and isread:
                nb = 193; Z = cap['z'].repeat(nb, 1).clone(); Z[1:] += eps_*torch.eye(192, dtype=T)
                ob, hb, _ = bstep_z(W, out.repeat(nb, 1), h.repeat(nb, 1, 1), dt, e[None].repeat(nb, 1), u[None].repeat(nb, 1), Mt.repeat(nb, 1, 1), ages, zfn=lambda a_, b_, c_, d_: Z, info=None)
                cc = ctx_next(W, ob, hb, dt2).numpy(); JJ_.append((float(info['hi'][0]), float(info['hn'][0]), (cc[1:]-cc[0]).T/eps_))
                if len(VG) < 3:
                    v0 = bview(W, out2, h2, dt2, zs=torch.zeros(1, 1, 32, dtype=T)); Vb = v0.repeat(193, *([1]*(v0.dim()-1))).clone()
                    Vb.reshape(193, -1)[1:, :192] += eps_*torch.eye(192, dtype=T); mm = []
                    for c in CODES4:
                        if c in reg: tl, zl, vl = reg[c]; hs_ = 1.0; ag = (nxt[0]-tl)*3600.0
                        else: hs_ = 0.0; ag = 0.0; zl = 0.0; vl = 0
                        p = bvalue_probs(W, Vb, c, torch.full((193,), hs_, dtype=T), torch.full((193,), ag, dtype=T), torch.full((193,), float(zl), dtype=T), torch.full((193,), int(vl), dtype=torch.long))
                        mm.append((p[:, 1:]*torch.tensor(CENd[c], dtype=T)).sum(1))
                    mm = torch.stack(mm, 1).numpy(); VG.append(((mm[1:]-mm[0])/eps_, tuple(v0.shape), float(np.abs(v0.reshape(-1)[:192].numpy()-c0).max())))
        out, h = out2, h2
    return JJ_, CS, VG
JX = []; CSa = []; VGa = []
for s in b1[:40]:
    a1_, a2_, a3_ = jac_ctx(Wt, BUN[s]); JX += a1_; CSa += a2_; VGa += a3_
print('drug bundles', len(JX), 'ctx samples', len(CSa), 'time', round(time.time()-t0, 1), 'view shape', VGa[0][1], 'view-ctx max diff', VGa[0][2])
Cr = np.stack([c for r_, _, c in CSa if r_]); S_c = np.cov(Cr.T); ev_s = np.linalg.eigvalsh(S_c); print('ctx(read) n', len(Cr), 'cov eig min/med/max', ev_s[0], np.median(ev_s), ev_s[-1])
Jm_c = np.mean([j for _, _, j in JX], 0)
for lamf in [1e-4, 1e-3, 1e-2]:
    Sr = S_c + lamf*np.mean(ev_s)*np.eye(192); Mg = Jm_c.T @ np.linalg.solve(Sr, Jm_c); ev_, evec_ = np.linalg.eigh(Mg); dstar = evec_[:, -1]
    wdec = np.linalg.solve(Sr, Jm_c @ dstar); g_k = np.array([wdec @ (j @ dstar) for _, _, j in JX]); nz_ = np.sqrt(wdec @ S_c @ wdec)
    print('lam', lamf, 'SNR per unit token norm (top 3)', np.round(np.sqrt(ev_[-3:][::-1]), 2), '| per-bundle gain mean/sd', round(float(g_k.mean()), 3), round(float(g_k.std()), 3), 'noise sd', round(float(nz_), 3))
Vg = np.mean([v[0] for v in VGa], 0); print('|grad view| per code', np.round(np.linalg.norm(Vg, axis=0), 3))
for nm, tgt in [('ins', np.array([0, -1., 0, -1.])), ('ne', np.array([1., 0, 1., 0]))]:
    cv = Vg @ np.linalg.solve(Vg.T @ Vg, tgt); print(nm, 'min-norm view shift for unit effect |c_v|', round(float(np.linalg.norm(cv)), 3), 'effect check', np.round(Vg.T @ cv, 3))
```