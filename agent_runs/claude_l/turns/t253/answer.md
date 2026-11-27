The SNR is poor. At the base after an infusion, the best token direction gives decode noise of 1.33 U/h per unit token norm for insulin and 1.55 for NE. The base moves only 0.04–0.06 per unit token change. Getting the noise below 0.1 U/h would need token perturbations about 15 times larger than the prelude can take linearly.

So the core's coupling from prelude to state is weak. Most of the base's natural variation comes from the readings memory through the blocks, which drowns the injected signal.

The weakest link may be downstream of the prelude output E rather than the prelude itself. If E has quiet directions the z token can move cleanly, I can add a strong channel there. A rank-1 edit to `core.b_in` would map E's quiet direction q onto a chosen state direction m with large gain. Only E's small natural variation along q would be amplified with it.

This run measures that E-level SNR: the Jacobian of E (4 slots × 192) with respect to z-token perturbations at about 390 drug bundles, against the natural covariance of E across bundles. Noise per unit token norm of about 0.1 U/h or better would make the b_in route viable. Noise near the base-level 1.3 would rule it out, leaving only the nonlinear direct-token regime.

```python
import numpy as np, torch, time
def E_of(W, h, dt, e, u, zt):
    hf = h*torch.exp(-dt*torch.exp(W['core.flow.a']))
    tok = torch.cat([hf+W['core.slots'], e[:, None], u[:, None], zt[:, None]], 1)
    tok = bself(W, tok, 'core.prelude.0'); return bln(W, tok[:, :4], 'core.prelude_norm')
def jac_E_stay(W, B_, D):
    times = [b[0] for b in B_]; nd = D.shape[0]
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]; res = []
    Dz = torch.cat([torch.zeros(1, 192, dtype=T), D], 0); Bt = nd+1; zb = mk_zfn('copy_drug', W)
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        inf = info_of(atoms) if ua else None
        if ua:
            Es = E_of(W, h.repeat(Bt, 1, 1), dt, e[None].repeat(Bt, 1), u[None].repeat(Bt, 1), u[None]+Dz).numpy()
            res.append((float(inf['hi']), float(inf['hn']), Es[0], (Es[1:]-Es[0])/eps_))
        out, h, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zb, info=inf)
    return res
t0c = time.time(); JE = []
for s in list(BUN)[:20]: JE += jac_E_stay(Wc, BUN[s], D_eps)
print('n', len(JE), 'time %.0fs' % (time.time()-t0c))
E0 = np.stack([r[2] for r in JE]); JEk = np.stack([r[3] for r in JE])   # (n,4,192), (n,192dirs,4,192)
ehi = np.array([r[0] for r in JE]); ehn = np.array([r[1] for r in JE])
for sl in range(4):
    SE = np.cov(E0[:, sl, :].T); SE += 1e-4*np.trace(SE)/192*np.eye(192); SEi = np.linalg.inv(SE)
    out_s = 'slot %d |E| %.1f natsd %.3f |' % (sl, np.linalg.norm(E0[:, sl], axis=1).mean(), np.sqrt(np.trace(SE)/192))
    for nm_, m in [('ins', (ehi>0)&(ehn==0)), ('ne', (ehn>0)&(ehi==0))]:
        Jm_ = JEk[m][:, :, sl, :].mean(0)    # (dirs, 192)
        lam, vec = np.linalg.eigh(Jm_ @ SEi @ Jm_.T); v1 = vec[:, -1]; g1 = Jm_.T @ v1; d1 = SEi @ g1
        gains = np.array([d1 @ (JEk[i][:, sl, :].T @ v1) for i in np.where(m)[0]]) / (d1 @ g1)
        out_s += ' %s noise %.3f gain sd %.2f |g1| %.3f' % (nm_, 1/np.sqrt(lam[-1]), gains.std(), np.linalg.norm(g1))
    print(out_s)
```