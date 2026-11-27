The norepinephrine policy is a titration rule:
- **Start:** always below a MAP of about 62, 82% at 62–65 and rare above 65. Starting rates vary widely (median 0.10).
- **Stop:** never below 65, about 19% of hours at MAP 75–90.
- **Titration:** the hourly rate ratio averages about 1.1–1.2 below 65, about 1.0 at 65–75 and about 0.83 above 75, with a large spread (log-SD ≈ 0.3).

That spread is the candidate random component for identifying norepinephrine's effect, if it is iid noise rather than a response to unrecorded information.

Next I'll pin down the titration form. I'll regress the log rate ratio on MAP, check whether arterial or cuff MAP drives it (or their mean), and examine the residual distribution, its autocorrelation, and whether it predicts earlier MAP trends.

```python
co = N[(N.ne_prev>0) & N['ne'].notna()].copy()
co['lr'] = np.log(co['ne']/co.ne_prev)
co['capped'] = (co['ne']>=1.0) | (co.ne_prev>=1.0)
c2 = co[~co.capped & (co.ne_prev>=0.05)]
for nm in ['ma','mc','mdec']:
    s = c2[c2[nm].notna()]
    # piecewise by MAP bins of 2.5 mmHg
    t = s.groupby((s[nm]//2.5)*2.5).lr.agg(['mean','std','size'])
    print(nm, t[(t['size']>300)].round(3).T.to_string())
both = c2[c2.ma.notna()&c2.mc.notna()]
for nm, x in [('ma',both.ma),('mc',both.mc),('avg',(both.ma+both.mc)/2)]:
    r,_,sd = ols(both.lr, [x], ['m'])
    print(nm, r.b.round(4).to_dict(), "resid sd", round(sd,4))
```