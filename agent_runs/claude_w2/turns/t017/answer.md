Glucose_finger and glucose_lab measure the same underlying glucose: they agree with no offset and differ with sd 9. A random +1 U/h of insulin at hour t changes later glucose by:

| hours later | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| effect (mg/dL) | −2.0 | −2.3 | −1.9 | −1.4 | −1.0 | −0.7 |

Lab readings show the same pattern. So the effect peaks about 2 hours after the dose and then fades by roughly ×0.72 per hour, a mean-reverting glucose with a short delay. These are total effects, which also include the clinicians' later dose reactions. That feedback should be small, since a 2 mg/dL shift rarely moves a patient across a band edge.

Next I'll estimate how strongly later insulin doses react to the random part e_t (the first stage F(j)). Then I'll deconvolve to get the fixed-dose impulse response K(k), which is what the "doses as recorded ×2" question needs. If F(j≥1) ≈ 0, K ≈ the reduced form R. I'll also check that NE doses and MAP don't respond to insulin noise, so that cross-effects don't contaminate the estimate.

```python
dd = gg[gg.ins.notna() & gg.gf.notna()].copy()
gsh = gg.groupby('stay_id')
for k in range(0,8):
    gg[f'ins_l{k}'] = gsh['ins0'].shift(-k)
    gg[f'nor_l{k}'] = gsh['nor0'].shift(-k)
dd = gg[gg.ins.notna() & gg.gf.notna() & (gg.gf>=185)].copy()
F = []; R = []; RN=[]; RM=[]
for k in range(0,8):
    s = dd[dd[f'ins_l{k}'].notna()]
    F.append(LinearRegression().fit(np.c_[s.e_ins, s.gf, s.h], s[f'ins_l{k}']).coef_[0])
    RN.append(LinearRegression().fit(np.c_[s.e_ins, s.gf, s.h], s[f'nor_l{k}']).coef_[0])
    if k>=1:
        s = dd[dd[f'gf_l{k}'].notna()]
        R.append(LinearRegression().fit(np.c_[s.e_ins, s.gf, s.h], s[f'gf_l{k}']).coef_[0])
        s = dd[dd[f'ma_l{k}'].notna() & dd.ma_ff.notna()]
        RM.append(LinearRegression().fit(np.c_[s.e_ins, s.gf, s.h, s.ma_ff], s[f'ma_l{k}']).coef_[0])
print('first stage F(j):', np.round(F,3))
print('NE response to e_ins:', np.round(RN,4))
print('MAP response to e_ins:', np.round(RM,3))
print('reduced form R(k), k=1..7:', np.round(R,3))
# deconvolution R(k) = sum_{j=0}^{k-1} F(j) K(k-j)
K = []
for k in range(1,8):
    val = R[k-1] - sum(F[j]*K[k-j-1] for j in range(1,k))
    K.append(val/F[0])
print('structural K(k):', np.round(K,3), 'sum', round(sum(K),2))
```