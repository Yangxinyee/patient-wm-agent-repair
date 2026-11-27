The rounding check rules out my diagnosis: rounding all weights costs only 1e-5. The constant category offsets in run 222 come from my comparison baseline. I measured those against the original model (r0), so they include E3b's own insulin effects: glucose rows +0.007, cuff rows −0.0065.

The real problem shows in the norepinephrine numbers. Cost on gated rows is non-monotonic: 0.035 at C = 1, 0.0165 at C = 2, 0.064 at C = 3. That points to bf16 rounding noise in the difference between unit a and unit b.
- The gate part κ·G has large weights, and the two units round it differently.
- That rounding noise, about 0.05 in pre-activation, is comparable to the small signal s at lam = 4.

The fix is a larger signal relative to the gate weights. With lam = 1 the signal is about 4x larger and stays within the gate margin.

Held-out cost at C = 2 already looks affordable: about +0.001 for norepinephrine on top of E3b.

This run measures the norepinephrine channel against E3b (rE) for lam 1 and 2 at C = 1.5, 2 and 2.5. It reports the ratio of rounding noise to signal, the slope and the bf16 bias, and a held-out cost at the chosen setting. Cost should now rise steadily with C, with clearly lower cost at a given C than before.

```python
catsE = lambda r: {nm: round(float((r[0]-rE[0])[m_].mean()),4) for nm,m_ in cats.items()}
for lam_ in [1.0, 2.0]:
    for C_ne in [1.5, 2.0, 2.5]:
        Wn = build_ne2(C_ne, wN, bN, kN, wf+wp, bf+bp, lam=lam_); r = eval_E(*Wn)
        sig = C_ne*10.0/(kN*lam_*Vref); sint = (WCd[:, gr, :]@(sig*wN)).mean(0)
        sact = (WCd[:, gr, :]@(Wn[0][124]-Wn[0][142])).mean(0)
        nr = np.std(sact - sint - (sact-sint).mean())/np.std(sint)
        sl = np.polyfit(DhN, (r[1]-rE[1])[gr], 1)[0]; rb = eval_E(*Wn, dt=torch.bfloat16)
        print('lam', lam_, 'C', C_ne, 'dAll vs E3b', round(float((r[0]-rE[0]).mean()),5), catsE(r), 'noise/sig', round(float(nr),3), 'slope', round(sl,2), 'bf16', round(float((rb[1]-r[1])[gr].mean()),3))
dho = np.zeros(len(sid))
for f in [0,1]:
    te_m = inA if f==0 else ~inA; tr_m = ~te_m; Xg = WCd[:, tr_m, :].reshape(-1,192)
    gwf, gbf = ridge_fit(Xg, np.tile(isA[tr_m].astype(float), nd), 1.0); gwp, gbp = ridge_fit(Xg, np.tile((ne1[tr_m]>0).astype(float), nd), 1.0)
    w_, b_ = fitdec(gr[tr_m[gr]], off[tr_m[off]], 0.1)
    r = eval_E(*build_ne2(2.0, w_, b_, kN, gwf+gwp, gbf+gbp, lam=1.0)); dho[te_m] = (r[0]-rE[0])[te_m]
print('heldout lam1 C2 vs E3b', round(float(dho.mean()),5), {nm: round(float(dho[m_].mean()),4) for nm,m_ in cats.items()})
```