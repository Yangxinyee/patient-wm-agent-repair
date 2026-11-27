The constrained directions fail. Once the cross-drug and persistence responses are nulled, very little lag-1 response is left (0.0065 per unit for insulin, 0.0024 for NE). Worse, the R² near 0 shows that per-query responses are almost entirely context-dependent scatter, not a consistent dose effect. An edit built that way would mostly add noise to factual predictions.

The right criterion is consistency per query, not the average coefficient. So I now fit the direction directly to the desired per-query pattern by least squares on the 48 basis responses:
- **Insulin:** finger response equal to the lag-1 insulin dose, MAP response zero, using queries with no recent NE.
- **NE:** MAP response equal to the lag-1 NE dose, finger response zero, using queries with no recent insulin.

A high fitted R² would show that a consistent dose effect is reachable through head 2. If it stays low, I need more injection dimensions (heads 0 and 1, with a decoder also zeroed on measurement tokens) or a different injection site.

```python
noN = (LN_[:, :4].sum(1)==0); noI = (LI[:, :4].sum(1)==0)
def fitdir(selF, yF, selA, yA, lam=1e-6):
    Xq = np.column_stack([DM[:, selF], DM[:, selA]]).T; y = np.concatenate([yF, yA])
    a = np.linalg.solve(Xq.T@Xq + lam*np.eye(48), Xq.T@y); f = Xq@a
    nF = selF.sum(); eF = f[:nF]-yF; eA = f[nF:]-yA
    return a, 1-(eF**2).sum()/(yF**2).sum(), np.sqrt((eA**2).mean()), np.linalg.norm(a), f[:nF]
sF = fm & noN; sA = am & noN
aI2, r2I, rmsA_I, nI, fI = fitdir(sF, LI[sF,0], sA, np.zeros(sA.sum()))
sl = np.linalg.lstsq(np.column_stack([LI[sF], np.ones(sF.sum())]), fI, rcond=None)[0]
print('INS dir: finger R2 %.3f art-leak rms %.4f |a| %.1f fitted slope lags %s' % (r2I, rmsA_I, nI, np.round(sl,3)))
sA2 = am & noI; sF2 = fm & noI
aN2, r2N, rmsF_N, nN, fN = fitdir(sA2, LN_[sA2,0], sF2, np.zeros(sF2.sum()))
sl = np.linalg.lstsq(np.column_stack([LN_[sA2], np.ones(sA2.sum())]), fN, rcond=None)[0]
print('NE dir: art R2 %.3f finger-leak rms %.4f |a| %.1f fitted slope lags %s' % (r2N, rmsF_N, nN, np.round(sl,3)))
print('n rows', sF.sum(), sA.sum(), sA2.sum(), sF2.sum(), '| ins lag1>0 in sF', (LI[sF,0]>0).sum(), 'ne lag1>0 in sA2', (LN_[sA2,0]>0).sum())
```