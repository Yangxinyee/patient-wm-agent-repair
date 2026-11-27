The norepinephrine level surprise has SD 0.13 mcg/kg/min and is essentially uncorrelated with glucose, hour and glucose_lab. It has small correlations, about -0.035, with the current and previous MAP. This is likely imperfection in my cells, since the step probably varies a little within a band, and adding MAP controls will absorb it. Map_cuff reads about 3 mmHg above map_arterial in the same hour, with SD 6.7 and correlation 0.9.

Next I will regress future map_arterial and map_cuff at horizons 1 to 12 on the surprise. The controls are the expected rate, previous rate, band, source, and the current and previous MAP. I will also regress future rates on the surprise, which persists because the rate is a random walk, and deconvolve to get the fixed-dose impulse response per mcg/kg/min-hour. A consistent positive response across map_arterial and map_cuff would mean a real pressor effect. A zero response would mean norepinephrine does nothing in this ICU.

```python
c = Cn.copy()
gc = P.groupby('stay_id')
for k in range(1,13):
    P[f'MA_p{k}']=gc['MA'].shift(-k); P[f'MC_p{k}']=gc['MC'].shift(-k); P[f'N0_p{k}']=gc['N0'].shift(-k)
P['N0_p0']=P.N0
c = c.join(P[[f'MA_p{k}' for k in range(1,13)]+[f'MC_p{k}' for k in range(1,13)]+[f'N0_p{k}' for k in range(0,13)]], rsuffix='_x')
c['expN'] = c.N0 - c.uN
c['mapv_f'] = c.mapv.fillna(c.MA_ff.fillna(c.MC_ff)); c['mapv_f']=c.mapv_f.fillna(75)
c['MA_m1f']=c.MA_m1.fillna(c.mapv_f)
for bnd in ['lo','mid','hi','none']: c[f'bd_{bnd}']=(c.band==bnd).astype(float)
def nreg(c, ycol):
    s = c[c[ycol].notna()]
    X = np.column_stack([np.ones(len(s)), s.expN, s.N_prev, s.mapv_f, s.MA_m1f, s.bd_lo, s.bd_hi, s.bd_none, (s.src=='MC').astype(float), s.uN])
    yv=s[ycol].values; b,*_=np.linalg.lstsq(X,yv,rcond=None); r=yv-X@b
    cov=np.linalg.pinv(X.T@X)*(r@r)/(len(s)-X.shape[1]); return b[-1], np.sqrt(cov[-1,-1]), len(s)
TMA=[];TMC=[];TN=[]
for k in range(1,13):
    a,sa,na=nreg(c,f'MA_p{k}'); m,sm,nm=nreg(c,f'MC_p{k}'); q,sq,_=nreg(c,f'N0_p{k-1}')
    TMA.append(a);TMC.append(m);TN.append(q)
    print(k, f"MA tot {a:.2f}±{sa:.2f} (n={na})  MC tot {m:.2f}±{sm:.2f}  N_t+{k-1}: {q:.3f}±{sq:.3f}")
IRA = deconv(np.array(TMA),np.array(TN)); IRC=deconv(np.array(TMC),np.array(TN))
print("MA IR per (mcg/kg/min)*1h:", np.round(IRA,2), "cum", np.round(np.cumsum(IRA),1))
print("MC IR:", np.round(IRC,2), "cum", np.round(np.cumsum(IRC),1))
```