The one-hour insulin effect per unit grows with glucose up to about 235, then falls back:

| glucose band | effect per unit (mg/dL) | per unit as % of mean glucose |
|---|---|---|
| ~158 | −1.45 | −0.92% |
| ~235 | −2.17 | −0.92% |
| ~276 | −1.74 | −0.63% |

The pooled interaction is weak (−0.003 per mg/dL, se 0.0015). So the effect is not cleanly proportional to glucose. The drop in the top band, where base doses are 4–6 units, could instead mean the response flattens at large doses. That matters a lot for ×2, which takes band-2 doses to about 4 and band-3 doses to about 6.

I will test curvature in dose directly, using the jitter, which spreads doses over about 0.5–2.5× base:
- within bands 1, 2 and 3 separately, regress the one-hour change on the jitter and the squared jitter;
- across bands, regress on dose and dose² with fixed effects for glucose.

A significant negative curvature would mean saturation, so the ×2 effect would be less than double and ×½ more than half. No curvature would support a linear response.

```python
for lo,hi,base in [(150,180,1),(180,220,2),(220,260,3),(260,400,4)]:
    sb = s1[(s1.glucose_finger>lo)&(s1.glucose_finger<=hi)].copy()
    X = np.column_stack([np.ones(len(sb)), sb.glucose_finger, sb.zj, sb.zj**2])
    bb = np.linalg.lstsq(X, sb.dG.values, rcond=None)[0]; rr = sb.dG.values-X@bb; se_=np.sqrt(np.diag(np.linalg.inv(X.T@X))*rr.var())
    print(f"band {lo}-{hi}: lin {bb[2]:.3f} ({se_[2]:.3f}) quad {bb[3]:.3f} ({se_[3]:.3f})  dose range {sb.insulin.quantile(.01)}-{sb.insulin.quantile(.99)}")
    # nonparametric by dose
    t = sb.groupby('insulin').agg(n=('dG','size'), dG=('dG','mean'), G=('glucose_finger','mean'))
    t = t[t.n>=100]
    print("   ", [(d, round(r.dG - 0.0*r.G,2), int(r.n)) for d, r in t.iterrows()])
s1['dose2'] = s1.insulin**2
b, se, n = fe_reg(s1, 'dG', ['insulin','dose2'], 'gbin')
print("pooled FE: coef dose", b[0].round(3), "(", se[0].round(3), ") dose^2", b[1].round(4), "(", se[1].round(4), ")")
```