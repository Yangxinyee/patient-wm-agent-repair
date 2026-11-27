The virtual last-value shift failed:
- the predicted mean moves only about 0.14 mg/dL per virtual mg/dL;
- the gain needed is about 10×, which gives noisy shifts (sd 9.3 against a target 3.6) and bin NLL worse by +0.02 (glucose) and +0.04 (MAP).

The fixed tilt failed because one global reference and unbounded tails blew up the far bins. The remedy is local, bounded tilts:
- four gated units, one per last-glucose-bin band (bins 1–6, 7–9, 10–12, 13–16);
- each unit is on only for glucose codes in its band. The gate is linear because with_code adds the code and last-bin embeddings;
- each tilt is centred on its band's mean prediction and clipped at ±2.5 predictive sd;
- the constant of each gate is cancelled in exactly the matching transition-table rows.

I'll test gains of 1.0 and 1.5 locally on factual bin NLL and open-loop response by arm, and save the edit if the NLL change is ≤ 0.

```python
order = np.argsort(imp); U4 = [int(u) for u in order[:4]]
print("imp 8 lowest", np.round(imp[order[:8]], 3))
arm_ = kd.arm.values; te = siD[i1] >= 50
grp = [(1, 6), (7, 9), (10, 12), (13, 16)]; gcod = [6, 10]
gates = []
for k, (lo, hi) in enumerate(grp):
    ing = np.isin(cdA, gcod) & (lbA >= lo) & (lbA <= hi); tgk = np.where(ing, 1.0, -1.0)
    dk = np.linalg.solve(Xa.T @ Xa / len(X) + 1e-8*np.eye(193), Xa.T @ tgk / len(X))
    rk = fa & ing; m_k, V_k = pmA[rk].mean(), VA[rk].mean(); sdk = np.sqrt(V_k)
    Tk = np.zeros(17); Tk[1:] = np.clip((c16 - m_k) / V_k, -2.5*sdk/V_k, 2.5*sdk/V_k)
    gates.append((dk, Tk, lo, hi)); print(k, "gate res max %.3f" % np.abs(Xa @ dk - tgk).max(), "m %.0f sd %.1f n %d" % (m_k, sdk, rk.sum()))
def build_g(lam, kap=4.0, G=12.0, w_=wr, badj=b_adj):
    W1n, b1n, W2n, b2n, Ttn = Wv1.copy(), bv1.copy(), Wv2.copy(), bv2.copy(), Tt.copy()
    for u in U4: b2n += Wv2[:, u] * H0[:, u].mean()
    for u, (dk, Tk, lo, hi) in zip(U4, gates):
        W1n[u] = G*dk[:192] + w_[:192]/kap; b1n[u] = G*dk[192] + (w_[192] + badj)/kap; W2n[:, u] = kap*lam*Tk
        for c in gcod:
            for lbv in range(lo, hi + 1): Ttn[c*17 + lbv] -= G*kap*lam*Tk
    return W1n, b1n, W2n, b2n, Ttn
for lam in [1.0, 1.5]:
    E = build_g(lam); PN = sm(gelu(X @ E[0].T + E[1]) @ E[2].T + E[3] + E[4][cdA*17 + lbA]); pmN = pmV(PN)[0]
    dn_ = -np.log(PN[r_, bidx] + 1e-12) + np.log(P0[r_, bidx] + 1e-12); ad = pmN - pmA
    rav = (pmN[i1] - pmN[i0]) - (pmA[i1] - pmA[i0])
    print(f"lam {lam}: fact shift sd {ad[fa].std():.2f} corr {np.corrcoef(ad[fa], s_c[fa])[0,1]:.2f} | dNLL glu {dn_[f0a & gl].mean():.4f} map {dn_[f0a & ~gl].mean():.4f} | resp",
          {a: (round(float(rav[te & (arm_ == a)].mean()), 2), round(float(dm[te & (arm_ == a)].mean()), 2)) for a in [1, 2, 3, 4, 6, 7]})
```