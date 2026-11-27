Run 252 confirmed the prelude computation exactly: keys are the first 192 rows of the kv weight and values the last 192, with zero reconstruction error. It also gave a mixed but usable picture of the amplifier.

- **Head 2 is the clean one.** On every drug bundle it puts at least 0.998 of its attention on u and z, split 50/50 between them, and at most 0.0016 on other tokens.
- **Head 1 is not clean.** On some drug bundles it attends elsewhere entirely.
- **The normalised u tokens span a very low-rank set.** Singular value ratio about 3×10⁻⁴ by component 40.
- **The normalised reading-e tokens are not exactly low-rank, but their tail is tiny.** Ratio about 3×10⁻⁴ near component 180. A direction b chosen in that tail would leak only about 0.03 in residual-equivalent units.

**Still to check:**
- whether head 2 puts any weight on slot tokens in reading bundles;
- how strongly a vector added to the slot streams at drug bundles moves the next predictions, and whether head 2's 48-dimension output range can move glucose without moving MAP, or MAP without moving glucose.

I'll write an explicit step that adds a vector c to the four slot streams after prelude attention, and check it against the exact step. Then I'll compute the 192-dimension Jacobian of the next reading's four predicted means with respect to c at drug bundles. The heads' output ranges then give the minimum-norm amplifier directions.

**What would confirm it:** a gradient with respect to c much larger than the z-token gradient of 0.1. I also need a modest head-2 direction that gives −1 on finger and lab and 0 on MAP, consistently across bundles, plus the same for norepinephrine on MAP.

```python
t0 = time.time()
def bstep_c(W, out, h, dt, e, u, Mt, ages, zt, cadd):
    B = out.shape[0]; hf = h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    tok = torch.cat([hf+W['core.slots'], e[:, None], u[:, None], zt[:, None]], 1)
    a = bln(W, tok, 'core.prelude.0.norm1'); q = a @ W['core.prelude.0.attn.q.weight'].T; kv = a @ W['core.prelude.0.attn.kv.weight'].T
    pr = torch.softmax(torch.einsum('bihd,bjhd->bhij', q.reshape(B, 7, 4, 48), kv[..., :192].reshape(B, 7, 4, 48))/math.sqrt(48), -1)
    x1 = tok + torch.einsum('bhij,bjhd->bihd', pr, kv[..., 192:].reshape(B, 7, 4, 48)).reshape(B, 7, 192) @ W['core.prelude.0.attn.out.weight'].T
    x1 = torch.cat([x1[:, :4]+cadd[:, None], x1[:, 4:]], 1); x2 = x1 + bmlp(W, bln(W, x1, 'core.prelude.0.norm3'), 'core.prelude.0.ffn')
    E = bln(W, x2[:, :4], 'core.prelude_norm'); ae = bmlp(W, torch.log1p(ages)[:, None], 'core.age'); Mk = Mt + ae[None]; x = hf + E
    for L in ['core.blocks.0.0', 'core.blocks.0.1']:
        a_ = bln(W, x, L+'.norm1'); x = x + bmha(W, a_, a_, L+'.attn'); x = x + bmha(W, bln(W, x, L+'.norm2'), Mk, L+'.cross'); x = x + bmlp(W, bln(W, x, L+'.norm3'), L+'.ffn')
    sp = F.softplus(W['core.delta_in']); abar = torch.exp(-sp*torch.exp(W['core.a_in'])); h2 = abar*hf + sp*(E @ W['core.b_in.weight'].T) + (x-(hf+E))
    y = bself(W, h2 @ W['core.c_out.weight'].T, 'core.coda.0'); return bln(W, y.mean(1), 'core.out_norm'), y
def pred4(W, ob, hb, dt2, reg, t2, nb):
    views = bview(W, ob, hb, dt2, zs=torch.zeros(nb, 1, 32, dtype=T)); mm = []
    for c in CODES4:
        if c in reg: tl, zl, vl = reg[c]; hs_ = 1.0; ag = (t2-tl)*3600.0
        else: hs_ = 0.0; ag = 0.0; zl = 0.0; vl = 0
        p = bvalue_probs(W, views, c, torch.full((nb,), hs_, dtype=T), torch.full((nb,), ag, dtype=T), torch.full((nb,), float(zl), dtype=T), torch.full((nb,), int(vl), dtype=torch.long))
        mm.append((p[:, 1:]*torch.tensor(CENd[c], dtype=T)).sum(1))
    return torch.stack(mm, 1).numpy()
def jac_c(W, B_, eps_=0.05, chk=False):
    times = [b[0] for b in B_]; out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; reg = {}; res = []; mx = 0.0
    for a in ea:
        if a['code'] in MEASC and a['kind'] == 0: reg[a['code']] = (times[0], a['num'][0], a['vb'])
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        for a in ea:
            if a['code'] in MEASC and a['kind'] == 0: reg[a['code']] = (t, a['num'][0], a['vb'])
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T); info = info_of(atoms) if ua else None
        cap = {}
        def zcap(c_, e_, u_, i_):
            zt = blin(W, bmlp(W, torch.cat([c_, e_, u_], -1), 'z_post')[:, :32], 'z_in'); cap['z'] = zt; return zt
        out2, h2, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zcap, info=info)
        if chk:
            oc, hc = bstep_c(W, out, h, dt, e[None], u[None], Mt, ages, cap['z'], torch.zeros(1, 192, dtype=T)); mx = max(mx, float((oc-out2).abs().max()), float((hc-h2).abs().max()))
        if ua and k+1 < len(B_):
            nxt = B_[k+1]; ea2, _ = split_fix(nxt[1])
            if any(a['code'] in MEASC and a['kind'] == 0 for a in ea2):
                nb = 193; C_ = torch.zeros(nb, 192, dtype=T); C_[1:] = eps_*torch.eye(192, dtype=T)
                ob, hb = bstep_c(W, out.repeat(nb, 1), h.repeat(nb, 1, 1), dt, e[None].repeat(nb, 1), u[None].repeat(nb, 1), Mt.repeat(nb, 1, 1), ages, cap['z'].repeat(nb, 1), C_)
                mm = pred4(W, ob, hb, (nxt[0]-t)*3600.0, reg, nxt[0], nb); res.append((float(info['hi'][0]), float(info['hn'][0]), (mm[1:]-mm[0])/eps_))
        out, h = out2, h2
    return res, mx
JC = []
for i, s in enumerate(b1[:20]):
    r_, mx = jac_c(Wt, BUN[s], chk=(i < 2)); JC += r_
    if i < 2: print('bstep_c check max err', mx)
print('bundles', len(JC), 'time', round(time.time()-t0, 1))
for hd in [1, 2]:
    rd = P_[read][:, hd]; dr = P_[drug][:, hd]
    print('head', hd, 'read: max w(slots)', round(float(rd[:, :, :4].max()), 4), 'p99', round(float(np.quantile(rd[:, :, :4].max(-1), .99)), 5), 'mean w(u+z)', round(float((rd[:, :, 5]+rd[:, :, 6]).mean()), 3), '| drug: max w(slots)', round(float(dr[:, :, :4].max()), 4), 'max w(e)', round(float(dr[:, :, 4].max()), 4))
Wo = W0['core.prelude.0.attn.out.weight'].numpy()
for nm, sel, tgt in [('ins', 0, np.array([0, -1., 0, -1.])), ('ne', 1, np.array([1., 0, 1., 0]))]:
    Jl = np.stack([r[2] for r in JC if r[sel] > 0.5]); Jm_ = Jl.mean(0)
    print(nm, 'n', len(Jl), '|grad wrt c| per code (cuff,finger,art,lab)', np.round(np.linalg.norm(Jm_, axis=0), 3))
    for hs in [[1], [2], [1, 2]]:
        R_ = np.hstack([Wo[:, 48*hh:48*hh+48] for hh in hs]); Jh = R_.T @ Jm_; a_ = Jh @ np.linalg.solve(Jh.T @ Jh, tgt); c_ = R_ @ a_
        eff = np.einsum('kic,i->kc', Jl, c_); print('   heads', hs, '|a|', round(float(np.linalg.norm(a_)), 3), '|c|', round(float(np.linalg.norm(c_)), 3), 'effect mean', np.round(eff.mean(0), 3), 'sd', np.round(eff.std(0), 3))
        globals()['Cdir_%s_%s' % (nm, ''.join(map(str, hs)))] = (a_, c_)
```