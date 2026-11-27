NE clearly raises MAP. Holding later doses fixed, a one-hour pulse of 1 mcg/kg/min raises MAP by about:

| hours later | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---|---|---|---|---|---|---|
| K_N(k), mmHg | +3.6 | +2.7 | +1.7 | +1.1 | +0.4 | +0.45 | ≈ 0 |

The total is about 10 mmHg per (mcg/kg/min · h), the same on arterial and cuff. The cuff reads 3 mmHg higher on average. Because the titration persists, the first stage decays only slowly (0.73, 0.56, 0.44, ...), so deconvolution was essential. NE may raise glucose slightly (+0.3 to +0.6, noisy), which is not scored.

Two things to check before I query the model:
1. Are both effects linear in dose level? This matters for ×2, ×3 and ×0 extrapolation, and for NE rates that would exceed the 1.0 cap.
2. Do the effects vary with patient state?

I'll estimate the 1-hour effect slope separately in low, mid and high rate bins for both drugs, and across MAP/glucose levels. A constant slope supports a linear impulse-response model of the truth. A falling slope at high rates would mean saturation.

```python
open('/work/notes.md','a').write("""
## NE effect estimate (runs 19-20)
- NE noise (rate - E[rate|prev rate, MAP band]) unpredictable (R2 0.001). Stops only from rate<=0.06 & MAP>=75.
- Structural K_N(k) on MAP per (mcg/kg/min * 1h): +3.6 +2.7 +1.7 +1.1 +0.4 +0.45 ~0 ... (sum ~10.4). Same on ma and mc. mc reads ~3 mmHg above ma (sd 6.7).
- NE first stage persists (0.73,0.56,0.44,...) due to random-walk titration. NE->glucose maybe +0.4 (noisy, not scored).
""")
# linearity checks: 1h effect by rate level
dn['rbin'] = pd.cut(dn.nor_prev, [0,0.1,0.2,0.35,0.5,0.7,1.01])
for rb, s in dn[dn.ma_l1.notna()].groupby('rbin', observed=True):
    lr = LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev], s.ma_l1)
    lr2 = LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev], s.ma_l2) if s.ma_l2.notna().all() else None
    s2 = s[s.ma_l2.notna()]; lr2 = LinearRegression().fit(np.c_[s2.e_nor, s2.map_pol, s2.nor_prev], s2.ma_l2)
    print('NE rate bin', rb, len(s), 'k1 slope', round(lr.coef_[0],2), 'k2', round(lr2.coef_[0],2), 'sd e', round(s.e_nor.std(),3))
dn['mapb'] = pd.cut(dn.map_pol, [0,55,65,75,200])
for mb, s in dn[dn.ma_l1.notna()].groupby('mapb', observed=True):
    lr = LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev], s.ma_l1)
    print('MAP band', mb, len(s), 'k1 slope', round(lr.coef_[0],2))
di = gg[gg.ins.notna() & gg.gf.notna() & gg.gf_l1.notna()].copy()
for (bd), s in di.groupby('band'):
    for lo,hi in [(0,1.25),(1.25,2.25),(2.25,3.25),(3.25,20)]:
        ss = s[(s.ins>=lo)&(s.ins<hi)]
        if len(ss)<2000: continue
        lr = LinearRegression().fit(np.c_[ss.ins, ss.gf], ss.gf_l1)
        print('ins band', bd, 'rate', lo, hi, len(ss), 'slope', round(lr.coef_[0],2))
```