Three findings from the last run:
- GD['map'] cleanly separates map_arterial from the other codes (margin about 1.8).
- GD['gf'] does not separate glucose_finger from glucose_lab, so the insulin gate needs a new direction.
- The stored rate reads have a noisy baseline on real gf rows: sd about 2 rate units. That would cost too much factual NLL, so the reads must be refit.

Next, I will compute head inputs for all 1,765 gf rows and for matching map_arterial rows, then:
- fit a gf-only gate orthogonal to the drug-signal directions, and validate it on fresh rows;
- fit minimum-variance rate reads (gain 1 on the own drug, 0 on the other);
- fix the tensor-to-numpy conversion and redo the activation and variance checks.

The fix works if the validated gate margin is positive and the read noise is below about 0.3 for insulin and 0.1 for norepinephrine.

```python
import numpy as np, torch
from scipy.special import erf
T2N = lambda t: (t.detach().double().numpy() if torch.is_tensor(t) else np.asarray(t, float))
rng = np.random.default_rng(1)
_wc0 = HD.wc; CODE = {'c': 'glucose_finger'}
HD.wc = lambda ctx, z, cn, *a, **k: _wc0(ctx, z, CODE['c'], *a, **k)
def qrows(df, cn, lvs):
    CODE['c'] = cn; qs, ts = [], []
    for (i, r), lv in zip(df.iterrows(), lvs):
        out = wc_full(int(r.prow), r.gap, float(lv), r.age_s); qs.append(tonp(out[0])); ts.append(tonp(out[-1]))
    return np.array(qs), np.array(ts)
try:
    QG, TG = qrows(GR, 'glucose_finger', GR.lv.values)
    QM, TM = qrows(GR, 'map_arterial', rng.uniform(40, 105, len(GR)))
    val = GR.sample(300, random_state=7).reset_index(drop=True)
    QV = {cn: qrows(val, cn, val.lv.values if cn == 'glucose_finger' else rng.uniform(VC[cn][0], VC[cn][-1], 300))[0] for cn in VC}
finally:
    del HD.wc
f6 = lambda x: float(np.round(x, 3))
dI, dN = DQ['ins'].mean(0), DQ['ne'].mean(0)
Pd = np.linalg.qr(np.stack([dI, dN], 1))[0]; proj = lambda X: X - (X @ Pd) @ Pd.T
Xg = np.vstack([Qc[cn] for cn in VC]); yg = np.concatenate([np.full(len(Qc[cn]), 1.0 if cn == 'glucose_finger' else -1.0) for cn in VC])
Xp = proj(Xg); mu_ = Xp.mean(0); A = Xp - mu_; AtA = A.T @ A
best = None
for lam in [1e-1, 1e-2, 1e-3]:
    w = np.linalg.solve(AtA + lam * np.trace(AtA) / 192 * np.eye(192), A.T @ (yg - yg.mean())); w = proj(w[None])[0]; c0 = yg.mean() - mu_ @ w
    sv = {cn: QV[cn] @ w + c0 for cn in VC}; oth = max(sv[cn].max() for cn in VC if cn != 'glucose_finger')
    mg_v = min(sv['glucose_finger'].min(), (QG @ w + c0).min()) - max(oth, (QM @ w + c0).max())
    print('gfgate lam', lam, 'val margin', f6(mg_v), 'gf min', f6((QG @ w + c0).min()), 'oth max', f6(oth), '|w|', f6(np.linalg.norm(w)), 'w.dI', f6(w @ dI), 'w.dN', f6(w @ dN))
    if best is None or mg_v > best[0]: best = (mg_v, w, c0, lam)
WG, CG = best[1], best[2]
def readdir(Q, d_own, d_oth, lam):
    C = np.cov(Q.T); C = C + lam * np.trace(C) / 192 * np.eye(192)
    D = np.stack([d_own, d_oth], 1); CiD = np.linalg.solve(C, D); return CiD @ np.linalg.solve(D.T @ CiD, np.array([1.0, 0.0]))
RR = {}
for d, Q, lvv in [('ins', QG, GR.lv.values), ('ne', QM, None)]:
    own, oth = (dI, dN) if d == 'ins' else (dN, dI); tr, va = np.arange(len(Q)) % 2 == 0, np.arange(len(Q)) % 2 == 1
    res = []
    for lam in [1e-2, 1e-3, 1e-4, 1e-5]:
        R = readdir(Q[tr], own, oth, lam); res.append((np.std(Q[va] @ R), lam))
        print('read', d, 'lam', lam, 'sd tr', f6(np.std(Q[tr] @ R)), 'sd va', f6(np.std(Q[va] @ R)), '|R|', f6(np.linalg.norm(R)), 'gain sd', f6(np.std(DQ[d] @ R)), 'cross', f6(np.mean(DQ['ne' if d == 'ins' else 'ins'] @ R)))
    sdv, lam = min(res, key=lambda t: t[0]); R = readdir(Q, own, oth, lam); RR[d] = (R, float(np.mean(Q @ R)), lam)
    if lvv is not None: print('corr with lv', f6(np.corrcoef(Q @ R, lvv)[0, 1]))
W0 = T2N(load_original('attr_heads.value_bin.0.weight')); b0 = T2N(load_original('attr_heads.value_bin.0.bias'))
W2v = T2N(load_original('attr_heads.value_bin.2.weight')); b2v = T2N(load_original('attr_heads.value_bin.2.bias'))
gelu = lambda x: 0.5 * x * (1 + erf(x / np.sqrt(2)))
o = RAW[0]; ref = tonp(HD.logits(HD.hidden(o[0]), o[-1]))
print('act check', f6(np.abs(ref - (gelu(Qc['glucose_finger'][0] @ W0.T + b0) @ W2v.T + b2v + Tc['glucose_finger'][0])).max()))
UN8 = [31, 74, 103, 114, 124, 171, 173, 181]
for cn, Q, Tt in [('glucose_finger', QG, TG), ('map_arterial', QM, TM)]:
    pre = Q @ W0.T + b0; lg = gelu(pre) @ W2v.T + b2v + Tt; p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    c = np.array(VC[cn]); pb = p[:, 1:] / p[:, 1:].sum(1, keepdims=True); m = pb @ c; var = (pb * (c[None] - m[:, None]) ** 2).sum(1)
    print(cn, 'p0', f6(p[:, 0].mean()), 'var q10/50/90', np.round(np.quantile(var, [.1, .5, .9]), 1), 'mean', f6(var.mean()), 'UN8 max pre', f6(pre[:, UN8].max()))
print('UN8 |W2 col|', np.round(np.linalg.norm(W2v[:, UN8], axis=0), 3), 'r0', {d: f6(RR[d][1]) for d in RR}, 'gate lam', best[3])
with open('/work/notes.md', 'a') as f:
    f.write("- Run 281: GD['gf'] does NOT separate gf from glucose_lab (gf min 2.02 < lab max 2.78); GD['map'] margin 1.78 OK. RD baselines noisy (gf sd 2.0 rate units) -> refit.\n"
            "- Run 282: QG/QM (1765 gf / map rows), QV val; WG/CG gf-only gate (orth to DQ dirs); RR[d]=(R, r0, lam) min-variance reads (own gain 1, cross 0).\n")
```