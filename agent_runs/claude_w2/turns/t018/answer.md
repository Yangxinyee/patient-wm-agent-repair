The run failed because I had only built leads up to 6 hours and the loop asked for lag 7. The plan is unchanged. I'll build leads out to 12 hours for all readings and doses, which I'll also need for the 12-hour horizon of the evaluation, and rerun the first-stage, reduced-form and deconvolution estimates.

```python
gsh = gg.groupby('stay_id')
for c in ['gf','gl','ma','mc']:
    for k in range(1,13):
        gg[f'{c}_l{k}'] = gsh[c].shift(-k)
for k in range(0,13):
    gg[f'ins_l{k}'] = gsh['ins0'].shift(-k)
    gg[f'nor_l{k}'] = gsh['nor0'].shift(-k)
dd = gg[gg.ins.notna() & gg.gf.notna() & (gg.gf>=185)].copy()
F = []; R = []; RN=[]; RM=[]
L = 10
for k in range(0,L+1):
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
print('reduced form R(k), k=1..:', np.round(R,3))
K = []
for k in range(1,L+1):
    val = R[k-1] - sum(F[j]*K[k-j-1] for j in range(1,k))
    K.append(val/F[0])
print('structural K(k):', np.round(K,3), 'sum', round(sum(K),2))
```