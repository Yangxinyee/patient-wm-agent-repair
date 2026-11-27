The head-side decoder fails. Even linear plus random features give negative R² for the factual residual. On edited rows R² is only 0.2–0.46 with slopes of 0.1–0.5, and predictions at bundles with no preceding drug have nearly the target's spread.

So the original model's state after an infusion barely carries the dose, let alone the residual. That fits its weak dose response and my earlier finding that the view decodes the residual with R² 0.30.

Two weak levers don't add up to a strong one, but they can be combined:
- **Write.** z_post writes r = rate − π̂ into the state through the z token at drug bundles, with a small amplitude so the prelude stays in its linear range.
- **Read.** z_prior reads it back from the next bundle's base and z_head amplifies it into the view, where the lever is strong.

Whether this works depends on the signal-to-noise ratio. It is set by the base Jacobian G, i.e. how a token perturbation moves the next bundle's base, against the natural spread Σ of that base.

This run measures G at about 390 drug bundles and Σ from the factual post-infusion bases I collected. It then finds the token direction with the best SNR, as the top eigenvector of G Σ⁻¹ Gᵀ, and reports:
- the decode noise per unit token perturbation;
- how consistent the decode gain is across bundles.

Noise well below 0.5 U/h at a token norm of about 0.5, with gain sd ≲ 0.25, would make the plan viable. Larger noise would put the 0.005-nat budget at risk.

```python
import numpy as np, torch, time
def jac_base_stay(W, B_, D):
    times = [b[0] for b in B_]; nd = D.shape[0]
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]
    res = []; Dz = torch.cat([torch.zeros(1, 192, dtype=T), D], 0); Bt = nd+1
    zb = mk_zfn('copy_drug', W); zp = mk_zfn('copy_drug', W, Dz)
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        inf = info_of(atoms) if ua else None
        if ua and k+1 < len(B_):
            o2, h2, _ = bstep_z(W, out.repeat(Bt, 1), h.repeat(Bt, 1, 1), dt, e[None].repeat(Bt, 1), u[None].repeat(Bt, 1), Mt.repeat(Bt, 1, 1), ages, zfn=zp, info=inf)
            bs = base_of(W, o2, h2, (times[k+1]-t)*3600.0).numpy()
            res.append((float(inf['hi']), float(inf['hn']), bs[0], (bs[1:]-bs[0])/eps_))
        out, h, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zb, info=inf)
    return res
t0c = time.time(); GBs = []
for s in list(BUN)[:20]: GBs += jac_base_stay(Wc, BUN[s], D_eps)
print('n', len(GBs), 'time %.0fs' % (time.time()-t0c))
Gk = np.stack([r[3] for r in GBs]); ghi = np.array([r[0] for r in GBs]); ghn = np.array([r[1] for r in GBs])
mF = (aft==1) & (fA==1); Sig = np.cov(Xb[mF].T); Sig += 1e-4*np.trace(Sig)/192*np.eye(192)
Si = np.linalg.inv(Sig); evS = np.linalg.eigvalsh(Sig); print('Sigma eig max %.3g min %.3g' % (evS.max(), evS.min()))
for nm_, m in [('ins-only', (ghi>0)&(ghn==0)), ('ne-only', (ghn>0)&(ghi==0)), ('both', (ghi>0)&(ghn>0)), ('all', np.ones(len(ghi), bool))]:
    Gm = Gk[m].mean(0); Mm = Gm @ Si @ Gm.T; lam, vec = np.linalg.eigh(Mm); v1 = vec[:, -1]
    g1 = Gm.T @ v1; d1 = Si @ g1; gains = np.array([d1 @ (Gk[i].T @ v1) for i in np.where(m)[0]]) / (d1 @ g1)
    print(nm_, 'n', m.sum(), 'noise per unit |v| %.3f (2nd %.3f, 5th %.3f)' % (1/np.sqrt(lam[-1]), 1/np.sqrt(lam[-2]), 1/np.sqrt(lam[-5])),
          'gain sd %.3f' % gains.std(), '|g1| %.3f' % np.linalg.norm(g1))
```