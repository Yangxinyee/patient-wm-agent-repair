The scan changes the picture. Ablating units 54 or 9 with global-mean compensation costs almost nothing: 0.00056 and 0.00002. Several units (124, 142, 14 …) are even slightly negative.

So the +0.011 on glucose rows in run 220 was not ablation. Most likely I rounded the entire head matrices to bf16 in `build_ne`, not only the entries I changed. That alters every unit and would also degrade the fp32 factual NLL.

This run will:
1. Check that rounding diagnosis directly.
2. Rebuild the norepinephrine channel on the cheap units 124 and 142, rounding only the modified entries.
3. Evaluate in-sample cost, slope and bf16 bias at C = 1, 2 and 3.
4. Get a held-out estimate at C = 2 from a 2-fold split by stay, where decoder and gates are fitted on the other half.

If rounding is the culprit, the off-gate category costs should fall to about 0. The C = 2 cost should then be roughly 0.001–0.002 overall.

```python
Wr = [bfr_(x) for x in (W1c,b1c,W2c,b2c)]; dr = eval_E(*Wr)[0]-rE[0]
print('round-all', round(float(dr.mean()),5), {nm: round(float(dr[m_].mean()),4) for nm,m_ in cats.items()}, 'exact?', [bool(np.array_equal(x, bfr_(x))) for x in (W1c,b1c,W2c,b2c)])
def build_ne2(C_ne, wd, bd, k, gw, gb, units=(124,142), kap=6.0, lam=4.0):
    ia, ib = units; W1n,b1n,W2n = W1c.copy(),b1c.copy(),W2c.copy()
    b2n = bfr_(b2c + W2c[:,ia]*mall[ia] + W2c[:,ib]*mall[ib])
    sig = C_ne*10.0/(k*lam*Vref); wg = kap*gw; bg = kap*(gb-1.5)
    W1n[ib] = bfr_(wg); b1n[ib] = bfr_(bg); W1n[ia] = bfr_(wg + sig*wd); b1n[ia] = bfr_(bg + sig*bd)
    W2n[:,ia] = bfr_(lam*tA); W2n[:,ib] = bfr_(-lam*tA)
    return W1n,b1n,W2n,b2n
kN = 0.194; wN, bN = fitdec(gr, off, 0.1); DhN = (WCd[:, gr, :]@wN + bN).mean(0)/kN
for C_ne in [1.0, 2.0, 3.0]:
    Wn = build_ne2(C_ne, wN, bN, kN, wf+wp, bf+bp); r = eval_E(*Wn); dn = r[0]-r0[0]
    sl = np.polyfit(DhN, (r[1]-rE[1])[gr], 1)[0]; rb = eval_E(*Wn, dt=torch.bfloat16)
    print('C', C_ne, 'dAll', round(float(dn.mean()),5), {nm: round(float(dn[m_].mean()),4) for nm,m_ in cats.items()}, 'slope', round(sl,2), 'bf16', round(float((rb[1]-r[1])[gr].mean()),3))
us = np.unique(sid); rng = np.random.default_rng(0); fA = set(rng.choice(us, len(us)//2, replace=False)); inA = np.isin(sid, list(fA))
dho = np.zeros(len(sid))
for f in [0,1]:
    te_m = inA if f==0 else ~inA; tr_m = ~te_m
    Xg = WCd[:, tr_m, :].reshape(-1,192)
    gwf, gbf = ridge_fit(Xg, np.tile(isA[tr_m].astype(float), nd), 1.0); gwp, gbp = ridge_fit(Xg, np.tile((ne1[tr_m]>0).astype(float), nd), 1.0)
    w_, b_ = fitdec(gr[tr_m[gr]], off[tr_m[off]], 0.1)
    r = eval_E(*build_ne2(2.0, w_, b_, kN, gwf+gwp, gbf+gbp)); dho[te_m] = (r[0]-r0[0])[te_m]
print('heldout C2 dAll', round(float(dho.mean()),5), {nm: round(float(dho[m_].mean()),4) for nm,m_ in cats.items()})
```