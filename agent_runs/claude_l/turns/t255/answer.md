Both neutral parts are cheap.
- **Copy of u:** a 28-component copy reproduces u to 0.2% on average and 1.6% at worst, including edited doses. At drug bundles it costs nothing measurable (dNLL −0.00016).
- **Reading-bundle token:** a constant t0 costs +0.00013 (mean of the original tokens) or +0.00016 (z_in.bias). That is about 3% of the 0.005 budget, so I'll use the mean token t0b.

Next I choose the injection pair: a token direction v and a quiet E direction q. The rank-2 b_in term will amplify only E's component along q. That component must not pick up any of the following:
- the mean of E at any bundle type, or it becomes a constant offset at every step;
- E's natural dependence on the dose through u, or it leaks a raw-dose effect, the very thing that costs factual NLL;
- natural fluctuations at reading bundles, where the term also fires.

So I'll collect E at all bundles of 30 tuning stays under the final baseline (copy at drug bundles, t0b at reading bundles). I project out the means and per-drug dose directions, and maximise SNR against drug- plus reading-bundle noise.

A confirming result is noise at or below about 0.2 per unit |v|, gain sd ≲ 0.4, and negligible dose leakage. If the constraints destroy the SNR, the b_in route fails.

```python
import numpy as np, torch
def zbase(ctx, e, u, info): return copyK(u) if info is not None else t0b.expand(u.shape[0], -1)
def collect_E(W, B_):
    times = [b[0] for b in B_]; res = []
    out = torch.tensor(X0row[:192], dtype=T)[None]; h = torch.tensor(X0row[192:960], dtype=T).reshape(1, 4, 192)
    ea, ua = split_fix(B_[0][1]); Ms = [set_enc(W, ea, 'encode')]; ts = [times[0]]
    for k in range(1, len(B_)):
        t = times[k]; dt = (t-times[k-1])*3600.0; atoms = B_[k][1]; ea, ua = split_fix(atoms)
        e = set_enc(W, ea, 'encode'); u = set_enc(W, ua, 'encode_action') if ua else u_empty
        Ms.append(e); ts.append(t); Mt = torch.stack(Ms)[None]; ages = torch.tensor([(t-tj)*3600.0 for tj in ts], dtype=T)
        inf = info_of(atoms) if ua else None
        Ek = E_of(W, h, dt, e[None], u[None], zbase(None, e[None], u[None], inf))[0].numpy()
        res.append((0 if inf is None else 1, 0.0 if inf is None else float(inf['hi']), 0.0 if inf is None else float(inf['hn']),
                    0.0 if inf is None else float(inf['di']), 0.0 if inf is None else float(inf['dn']), Ek))
        out, h, _ = bstep_z(W, out, h, dt, e[None], u[None], Mt, ages, zfn=zbase, info=inf)
    return res
EC = []
for s in list(BUN)[:30]: EC += collect_E(Wc, BUN[s])
isd = np.array([r[0] for r in EC]); chi = np.array([r[1] for r in EC]); chn = np.array([r[2] for r in EC])
cdi = np.array([r[3] for r in EC]); cdn = np.array([r[4] for r in EC]); EE = np.stack([r[5] for r in EC])   # (n,4,192)
types = {'read': isd==0, 'ins': (isd==1)&(chi>0)&(chn==0), 'ne': (isd==1)&(chn>0)&(chi==0), 'both': (isd==1)&(chi>0)&(chn>0)}
cons_ = [EE[m].mean((0, 1)) for m in types.values()]
# dose directions via regression of slot-avg E on rate within type
Ea = EE.mean(1)
for nm_, m, dd in [('ins', types['ins'], cdi), ('ne', types['ne'], cdn), ('bi', types['both'], cdi), ('bn', types['both'], cdn)]:
    X_ = np.c_[dd[m], np.ones(m.sum())]; coef = np.linalg.lstsq(X_, Ea[m], rcond=None)[0][0]; cons_.append(coef)
Cmat = np.stack(cons_); Qc, _ = np.linalg.qr(Cmat.T); Bc = np.linalg.svd(np.eye(192) - Qc @ Qc.T)[0][:, :192-Cmat.shape[0]]
# noise: residual E variation (after type means) pooled over slots, drug + reading
resid = []
for nm_, m in types.items():
    R_ = EE[m] - EE[m].mean(0, keepdims=True); resid.append(R_.reshape(-1, 192))
Sn = np.cov(np.vstack(resid).T); Sn += 1e-4*np.trace(Sn)/192*np.eye(192)
SnB = Bc.T @ Sn @ Bc; SnBi = np.linalg.inv(SnB)
VQ = {}
for nm_, m in [('ins', (ehi>0)&(ehn==0)), ('ne', (ehn>0)&(ehi==0))]:
    Jm_ = JEk[m].mean((0, 2))   # (dirs, 192) slot-averaged
    JB = Jm_ @ Bc; lam, vec = np.linalg.eigh(JB @ SnBi @ JB.T); v1 = vec[:, -1]
    a1 = SnBi @ (JB.T @ v1); q1 = Bc @ a1; q1 = q1 / (q1 @ (Jm_.T @ v1))     # unit gain per unit token along v1
    gains = np.array([q1 @ (JEk[i].mean(1).T @ v1) for i in np.where(m)[0]])
    nz = {k: np.std(EE[mm].reshape(-1, 192) @ q1) for k, mm in types.items()}
    dl = [abs(q1 @ c) for c in cons_[4:]]
    VQ[nm_] = (v1, q1)
    print(nm_, 'noise/unit|v| %.3f' % (1/np.sqrt(lam[-1])), 'gain mean %.2f sd %.2f' % (gains.mean(), gains.std()),
          'nat q.E sd by type', {k: round(float(x), 3) for k, x in nz.items()}, 'dose leak', np.round(dl, 4).tolist(),
          'mean offsets', np.round([q1 @ c for c in cons_[:4]], 4).tolist())
```