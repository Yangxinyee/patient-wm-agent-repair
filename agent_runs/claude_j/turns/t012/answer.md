The rate multiplier behaves like pure noise. Its log has SD about 0.31 at every base level, with no autocorrelation (0.004). The variance of stay means matches what i.i.d. noise would give, and correlations with glucose, MAP, norepinephrine and hour are at most 0.015. So the clinician's within-band dose scatter is effectively randomised: an instrument far more powerful than the RD.

Next I will define the dose surprise u = rate − mean rate for the band and regress future glucose_finger at horizons 1 to 10 on u. Band dummies and the current and previous glucose serve only as precision controls. I will also regress future insulin on u to measure the protocol feedback, then deconvolve to get the fixed-dose impulse response. An impulse response matching the RD (about -2 for 3 hours, then decay) with much smaller standard errors would confirm it. A persistent tail would change the expected readout considerably.

```python
mean_by_base = s.groupby('base').I.mean()
print(mean_by_base)
P['u'] = np.where(P.I.notna()&(P.base>0), P.I - P.base.map(mean_by_base), np.nan)
P['GF_m1'] = g['GF'].shift(1)
dd = P[P.u.notna() & (P.h<=23)].copy()
for b in [1,2,3,4,6]: dd[f'b{b}'] = (dd.base==b).astype(float)
def surprise_reg(dd, ycol, extra=['GF','GF_m1']):
    s = dd[dd[ycol].notna()].copy()
    s['GF_m1f'] = s.GF_m1.fillna(s.GF); s['hasm1']=s.GF_m1.notna().astype(float)
    X = np.column_stack([s[f'b{b}'] for b in [1,2,3,4,6]] + [s.GF, s.GF_m1f, s.hasm1, s.u])
    yv = s[ycol].values
    b,*_ = np.linalg.lstsq(X, yv, rcond=None)
    r = yv - X@b
    cov = np.linalg.inv(X.T@X)*(r@r)/(len(s)-X.shape[1])
    return b[-1], np.sqrt(cov[-1,-1]), len(s)
for k in range(9,13): P[f'GF_p{k}'] = g['GF'].shift(-k)
for j in range(9,13): P[f'I0_p{j}'] = g['I0'].shift(-j)
dd = P[P.u.notna()].copy()
for b in [1,2,3,4,6]: dd[f'b{b}'] = (dd.base==b).astype(float)
TY=[];TI=[]
for k in range(1,13):
    a,se,n = surprise_reg(dd, f'GF_p{k}')
    c,sc,_ = surprise_reg(dd, f'I0_p{k-1}')
    TY.append(a); TI.append(c)
    print(k, "GF tot:", round(a,3), "±", round(se,3), "n=",n, " I_{t+%d}:"%(k-1), round(c,3), "±", round(sc,3))
IRs = deconv(np.array(TY), np.array(TI))
print("deconvolved IR:", np.round(IRs,2), "cum:", np.round(np.cumsum(IRs),2))
```