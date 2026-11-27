Dose information reaches the latent z but barely reaches the recurrent state:
- In z, the dose explains 14% of the residual variance at insulin bundles (|b| = 0.80 per U/h, residual rms 0.20 per dimension).
- In the change of out across the drug bundle it explains 0.3%, and in the change of the h slots 0.2%.

So the dose is encoded in z, but little of it gets into h and out, which the heads read. That is consistent with the near-zero teacher-forced dose response. z is the natural channel to amplify.

Next I will characterise the z dose code:
1. Does it encode the raw dose D or the deviation D − base? I will replace the tier dummies with base and a smooth glucose term, so the D and base coefficients separate.
2. Is the dose direction in z drug-specific? I will compare the spread of z along that direction at reading bundles, NE bundles and insulin bundles.
3. How large is the posterior mean along it?

A drug-specific deviation code would make z a safe place to edit.

```python
bD_z = None
Xc = [np.ones(len(ib)), g, g**2/100, base, D]
Bz, Rz = multi_ols(Z, Xc)
bD = Bz[-1]; bB = Bz[-2]
dz = bD/np.linalg.norm(bD)
print("cos(bD, -bBase) = %.3f; |bD|=%.3f |bBase|=%.3f" % (dz@(-bB)/np.linalg.norm(bB), np.linalg.norm(bD), np.linalg.norm(bB)))
proj = X[:,960:992]@dz
st2['zproj'] = proj
st2['kind'] = np.where(st2.frac==0.0833, np.where(st2['ins']&~st2['ne'],'ins', np.where(st2['ne']&~st2['ins'],'ne', np.where(st2['ins']&st2['ne'],'both','end'))), np.where(st2.frac==0,'reading','other'))
print(st2.groupby('kind').zproj.agg(['size','mean','std']).round(3))
# relation of projection with dose in insulin bundles
x = st2[st2.kind=='ins']
print("corr(zproj, D) ins-only:", np.corrcoef(x.zproj, x.insulin)[0,1])
xb = st2[st2.kind=='both']
print("corr(zproj, D) both:", np.corrcoef(xb.zproj, xb.insulin)[0,1], " corr with NE:", np.corrcoef(xb.zproj, xb.norepinephrine)[0,1])
xn = st2[st2.kind=='ne']
print("corr(zproj, NE) ne-only:", np.corrcoef(xn.zproj, xn.norepinephrine)[0,1])
```