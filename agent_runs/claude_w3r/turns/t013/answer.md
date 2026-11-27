The IV structural response is −1.70, −1.99, −1.61, −1.15, −0.67, −0.23 at lags 1–6, then turns positive (+0.39, +0.52) at lags 7–8, with SE about 0.1. The positive tail is suspicious. In a simple mean-reverting system it would need a rebound. It also clashes with the reduced form (−0.24, −0.12 at 7–8), since feedback through lower later insulin should push the reduced form above, not below, the structural values.

To check this, next I will directly measure how an insulin deviation at t changes later insulin r_{t+m}, i.e. the policy feedback. I will then reconstruct the structural response recursively from the reduced form: reduced(s) = Σ h(s−m)·dr(m). If dr(m) is small or negative, the structural and reduced-form responses should nearly agree, and the IV tail is an artefact.

```python
res={}
for s in range(0,9):
    rows=[]
    for t in range(1,T-s):
        rows.append(pd.DataFrame({'g':gf[:,t],'y':ins_r[:,t+s],'yg':gf[:,t+s] if s>0 else np.nan,'x':xdev[:,t],'stay':np.arange(n)}))
    F=pd.concat(rows).dropna(subset=['g','x']); F['g2']=(F.g-160)**2/100
    b,s_,nn=ols(F,'y',['x','g','g2']); res[s]=b.x
    print('m=%d effect of x_t on r_{t+m}: %.4f se %.4f'%(s,b.x,s_.x))
red = np.array([-1.971,-2.230,-1.743,-1.194,-0.794,-0.487,-0.235,-0.123])
dr = np.array([res[m] for m in range(9)])
h=np.zeros(9)
for s in range(1,9):
    h[s]=(red[s-1]-sum(h[s-m]*dr[m] for m in range(1,s)))/dr[0]
print('recursive structural h:', np.round(h[1:],3))
```