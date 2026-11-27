Norepinephrine raises arterial MAP. Per 1 mcg/kg/min at hour s, with all other doses held fixed, the response is:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7–12 |
|---|---|---|---|---|---|---|---|
| mmHg | +3.16 | +2.63 | +1.78 | +0.95 | +0.46 | +0.33 | ≈ 0 |

Standard errors are 0.11–0.26. A sustained +1 mcg/kg/min gives a steady state of about +9.4 mmHg, so doubling a typical 0.5 should raise MAP by roughly 4–5 mmHg at plateau. The model currently answers −0.08 (x2) and −0.48 (x1/2), which is essentially no response and has the wrong sign for x2.

Next I will:
- compute the implied true norepinephrine readouts on the 100 tuning stays at the tool's reached map_arterial hours;
- run placebo tests on both instruments: the dose residual at s must not predict readings at s−1 or s, which came before the dose.

Passing placebos would support the instruments' validity. Correlation with pre-dose readings would mean the policy model misses something and the estimates are biased.

```python
json.dump({'beta_ins': beta_ins.tolist(), 'gam_ne': gam.tolist()}, open('/work/effects.json','w'))
def true_shift_ne(sid, hour, factor, gam_):
    i = sid2i[sid]; t = hour+1
    dN = NOR_U[i]*(factor-1); dN[:13]=0
    return sum(gam_[k]*dN[t-k] for k in range(1, min(Kn, t-13)+1) if t-k>=13)
for e_, f_ in [('ne_x2',2),('ne_half',0.5)]:
    m_ = de.edit==e_
    de.loc[m_,'true'] = [true_shift_ne(s,h,f_,gam) for s,h in zip(de.stay_id[m_], de.hour[m_])]
r = de[de.reached]
print(r.groupby('edit')[['shift','true']].mean())
print(r[r.edit=='ne_x2'].groupby('hour')[['shift','true']].mean().round(2).T.to_string())
# placebo tests: residual instrument vs pre-dose readings
GFm1 = np.full_like(GF,np.nan); GFm1[:,1:] = GF[:,:-1]
sel = GIVEN & ~np.isnan(GFm1) & ~np.isnan(GF)
X_ = np.c_[GF[sel], BASE[sel]==1, BASE[sel]==2, BASE[sel]==3]
r1 = GFm1[sel] - LinearRegression().fit(X_, GFm1[sel]).predict(X_)
print('placebo ins: corr(eps_s, resid gf_{s-1})', np.corrcoef(EPS[sel], r1)[0,1], 'n', sel.sum())
MAm1 = np.full_like(MA,np.nan); MAm1[:,1:]=MA[:,:-1]
sel = cont & ~np.isnan(MA) & ~np.isnan(MAm1)
X_ = np.c_[MA[sel], NP[sel], NP[sel]**2]
r2 = MAm1[sel]-LinearRegression().fit(X_, MAm1[sel]).predict(X_)
print('placebo ne: corr(eps_s, resid ma_{s-1})', np.corrcoef(EPSN[sel], r2)[0,1], 'n', sel.sum())
print('corr(eps_ne, ma_s) raw', np.corrcoef(EPSN[sel], MA[sel])[0,1])
```