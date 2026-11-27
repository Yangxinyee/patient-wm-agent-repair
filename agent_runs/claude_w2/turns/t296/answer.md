The run only printed the tensor names. The value head is `attr_heads.value_bin` (MLP 192→192→17) plus `embed_transition.value_bin.weight` (221 = 13 codes × 17 last-bins).

Now the actual build:
- **Gated insulin readout:** fitted on the arm differences (μ = 0.3) and centred on factual rows.
- **Unit placement:** it goes into the least important hidden unit, with G = 20, κ = 4 and gain 1.5.
- **Old unit:** its mean contribution moves into the output bias.
- **Compensation:** the transition table offsets the gate's constant on the glucose rows.

Local checks on `D4` rows:
1. pre-activations stay in GELU's linear range on glucose rows and stay off on MAP rows;
2. the gate's residual error is small;
3. the open-loop shift per arm compares sensibly with the target;
4. the factual bin NLL for glucose and MAP does not get worse.

Then I'll save the edit and record it in the notebook.

```python
import numpy as np, torch, torch.nn.functional as F
def L64(n): return np.asarray(load(n), dtype=np.float64)
print("attr_heads:", [n for n in NAMES if n.startswith('attr_heads') and n.endswith('2.weight')])
Wv1, bv1, Wv2, bv2, Tt = [L64(n) for n in ['attr_heads.value_bin.0.weight','attr_heads.value_bin.0.bias','attr_heads.value_bin.2.weight','attr_heads.value_bin.2.bias','embed_transition.value_bin.weight']]
gelu = lambda a: np.asarray(F.gelu(torch.as_tensor(a)))
cdA = np.asarray(D4['code']).astype(int); lbA = np.asarray(D4['lb']).astype(int).ravel()
def logits_(W1, b1, W2, b2, T_):
    H = gelu(X @ W1.T + b1); return H @ W2.T + b2 + T_[cdA*17 + lbA], H
def sm(l): e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
cen = cenA[:, :16]
def pmV(P):
    p = P[:, 1:]; s_ = p.sum(1); m_ = (p*cen).sum(1)/s_; return m_, (p*(cen-m_[:, None])**2).sum(1)/s_
lg0, H0 = logits_(Wv1, bv1, Wv2, bv2, Tt); P0 = sm(lg0); pmA, VA = pmV(P0)
print("pm(zmu) vs stored pm: mean abs diff %.2f" % np.abs(pmA - pm_)[gl].mean())
imp = H0.std(0) * np.linalg.norm(Wv2, axis=0); order = np.argsort(imp); jg, jn = int(order[0]), int(order[1])
print("units", jg, jn, "imp", np.round(imp[order[:3]], 4), "median", round(float(np.median(imp)), 4))
Xa = np.c_[X, np.ones(len(X))]; tg = np.where(np.isin(cdA, [6, 10]), 1.0, -1.0)
dg = np.linalg.solve(Xa.T @ Xa / len(X) + 1e-8*np.eye(193), Xa.T @ tg / len(X)); eps = Xa @ dg - tg
print("gate resid max %.4f sd %.5f" % (np.abs(eps).max(), eps.std()))
wr = np.linalg.solve(Gd + 0.3*Gf + np.diag(np.r_[1e-3*np.ones(192), 1e-6]), hd + 0.3*hf)
s = Xa @ wr; b_adj = -s[fa].mean(); s_c = s + b_adj
kap, G, lam = 4.0, 20.0, 1.5
pre = G*(Xa @ dg) + s_c/kap
print("pre glucose min %.2f | MAP max %.2f" % (pre[gl].min(), pre[~gl].max()))
c16 = cen[cdA == 6][0]; print("gf/gl centres equal:", np.allclose(c16, cen[cdA == 10][0]))
Vbar, cref = VA[fa].mean(), pmA[fa].mean(); Tv = np.zeros(17); Tv[1:] = (c16 - cref)/Vbar
W1n, b1n, W2n, b2n, Ttn = Wv1.copy(), bv1.copy(), Wv2.copy(), bv2.copy(), Tt.copy()
b2n += Wv2[:, jg] * H0[:, jg].mean()
W1n[jg] = G*dg[:192] + wr[:192]/kap; b1n[jg] = G*dg[192] + (wr[192] + b_adj)/kap
W2n[:, jg] = kap*lam*Tv
for c in [6, 10]: Ttn[c*17:(c+1)*17] -= G*kap*lam*Tv
lg1, _ = logits_(W1n, b1n, W2n, b2n, Ttn); P1 = sm(lg1); pm1, _ = pmV(P1)
add = (pm1 - pmA)
print("glucose factual shift: mean %.2f sd %.2f | target lam*s sd %.2f" % (add[fa].mean(), add[fa].std(), lam*s_c[fa].std()))
ra = (pm1[i1]-pm1[i0]) - (pmA[i1]-pmA[i0])
print("open-loop added resp by arm (test) vs target dm:", {a: (round(float(ra[te & (arm_ == a)].mean()), 2), round(float(dm[te & (arm_ == a)].mean()), 2)) for a in [1, 2, 3, 4, 6, 7]})
valA = np.asarray(D4['val'], dtype=float); bidx = 1 + np.argmin(np.abs(valA[:, None] - cen), 1); r_ = np.arange(len(valA))
f0a = (armD == 0) & np.isfinite(valA)
dn = -np.log(P1[r_, bidx] + 1e-12) + np.log(P0[r_, bidx] + 1e-12)
print("factual coarse dNLL: glucose %.4f  MAP %.4f  all %.4f" % (dn[f0a & gl].mean(), dn[f0a & ~gl].mean(), dn[f0a].mean()))
EDIT_G = dict(jg=jg, jn=jn, kap=kap, G=G, lam=lam, Tv=Tv, dg=dg, wr=wr, b_adj=b_adj)
```