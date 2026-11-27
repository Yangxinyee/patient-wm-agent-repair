The head computes with_code = q + code embedding + last_numeric(f8) + the last-bin embedding + .... The z-scored last value enters linearly through `last_numeric.weight[:, 6]` and also through the last-bin embedding.

A virtual shift of the last glucose value is therefore the input direction
vdir = last_numeric[:, 6]/iqr + (mean adjacent-bin embedding difference)/bin-width.

The edit becomes a rank-1 change of the value MLP's first layer: x → x + g·s(x)·vdir. The readout s is refitted to be ≈0 on MAP rows, which provides the gate.

This run will:
1. measure how much the predicted mean moves per virtual mg/dL, and how linear that is for ±10 and ±20;
2. build the edit with gain 1.5;
3. check factual bin NLL for glucose and MAP, and the open-loop response by arm.

If the factual NLL change is no worse than zero and the response is proportionate, I'll save the edit.

```python
Wt = {n: torch.as_tensor(L64(n)) for n in ['last_numeric.weight', 'embed.bins.value.weight']}
u_num = Wt['last_numeric.weight'][:, 6].numpy(); Eb = Wt['embed.bins.value.weight'].numpy()
dE = ((Eb[3:17] - Eb[1:15]) / 2).mean(0)
gv = valA[(cdA == 6) & np.isfinite(valA)]; iqr_g = np.subtract(*np.percentile(gv, [75, 25])); bw = (c16[-2] - c16[1]) / 13
print("iqr_g est %.1f  binw %.1f" % (iqr_g, bw))
vdir = u_num / iqr_g + dE / bw
def pm_of(Xin, W1, b1, W2=Wv2, b2=bv2, T_=Tt):
    H = gelu(Xin @ W1.T + b1); return pmV(sm(H @ W2.T + b2 + T_[cdA*17 + lbA]))[0]
gm_ = fa  # factual glucose rows
for eps_ in [-20, -10, 10, 20]:
    d_ = pm_of(X[gm_] + eps_ * vdir, Wv1, bv1) if False else None
Xg = X.copy()
sens = {}
for eps_ in [-20, -10, 10, 20]:
    Xs = X + eps_ * vdir; H = gelu(Xs @ Wv1.T + bv1); pmS = pmV(sm(H @ Wv2.T + bv2 + Tt[cdA*17 + lbA]))[0]
    sens[eps_] = round(float(((pmS - pmA)[fa]).mean() / eps_), 3)
print("pm sensitivity per virtual mg/dL:", sens)
kv = np.mean([sens[10], sens[-10]])
mapr = (armD == 0) & ~gl
Gm = Xa[mapr].T @ Xa[mapr] / mapr.sum()
for nu in [1.0]:
    wr2 = np.linalg.solve(Gd + 0.3*Gf + nu*Gm + np.diag(np.r_[1e-3*np.ones(192), 1e-6]), hd + 0.3*hf)
s2 = Xa @ wr2; b2adj = -s2[fa].mean(); wr2[192] += b2adj; s2 = Xa @ wr2
print("readout: glucose fact sd %.2f | MAP rows sd %.2f mean %.2f" % (s2[fa].std(), s2[mapr].std(), s2[mapr].mean()))
lam = 1.5; gg_ = lam / kv
a_vec = Wv1 @ vdir
W1v = Wv1 + gg_ * np.outer(a_vec, wr2[:192]); b1v = bv1 + gg_ * a_vec * wr2[192]
pmN = pmV(sm(gelu(X @ W1v.T + b1v) @ Wv2.T + bv2 + Tt[cdA*17 + lbA]))[0]
addv = pmN - pmA
print("factual shift glucose: mean %.2f sd %.2f (target sd %.2f, corr %.2f)" % (addv[fa].mean(), addv[fa].std(), lam*s2[fa].std(), np.corrcoef(addv[fa], s2[fa])[0, 1]))
PN = sm(gelu(X @ W1v.T + b1v) @ Wv2.T + bv2 + Tt[cdA*17 + lbA])
dnv = -np.log(PN[r_, bidx] + 1e-12) + np.log(P0[r_, bidx] + 1e-12)
print("factual coarse dNLL: glucose %.4f  MAP %.4f  all %.4f" % (dnv[f0a & gl].mean(), dnv[f0a & ~gl].mean(), dnv[f0a].mean()))
rav = (pmN[i1]-pmN[i0]) - (pmA[i1]-pmA[i0])
print("open-loop resp by arm (test) vs target:", {a: (round(float(rav[te & (arm_ == a)].mean()), 2), round(float(dm[te & (arm_ == a)].mean()), 2)) for a in [1, 2, 3, 4, 6, 7]})
```