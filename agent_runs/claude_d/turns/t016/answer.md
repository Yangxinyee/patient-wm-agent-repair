Norepinephrine is titrated multiplicatively. rate_t ≈ rate_{t−1} × f(MAP_t) × noise, with f about 1.15 below MAP 65, about 1.0 at 65–75 and about 0.85 at 75 and above. The noise is large (log SD ≈ 0.29) and does not change with MAP, the rate is capped at 1.0, and infusions stop only once the rate drifts below about 0.08.

So, given MAP_t and the previous rate, the hourly noise is a natural experiment for norepinephrine. Because the rate is persistent, past noise builds up in the rate level, which future MAP may also respond to.

Next I will estimate the norepinephrine effect: next-hour arterial MAP regressed on the current rate, controlling for current MAP and the previous rate, then at longer horizons. I will use the same layout as the insulin analysis to see whether the effect is immediate and level-based or builds up.

```python
for k in [1,2,3]:
    c[f'ma_p{k}'] = gg.ma.shift(-k); c[f'nor_p{k}']=gg.nor0.shift(-k)
dn = c[(c.nor_prev>0)&c.nor.notna()&c.ma.notna()].copy()
for k in [1,2,3]:
    dd = dn[dn[f'ma_p{k}'].notna()]
    X = np.c_[dd.ma, dd.nor_prev, dd.nor]
    lr = LinearRegression().fit(X, dd[f'ma_p{k}']); print('k',k,'n',len(dd),'coef ma, nor_prev, nor:', np.round(lr.coef_,3))
    X2 = np.c_[dd.ma, np.log(dd.nor_prev), np.log(dd.nor)]
    lr2 = LinearRegression().fit(X2, dd[f'ma_p{k}']); print('   log version coef:', np.round(lr2.coef_,3))
# nonparametric: within ma bin and prev-rate bin, relation of ma_p1 to ratio
dd = dn[dn.ma_p1.notna()].copy(); dd['lr']=np.log(dd.nor/dd.nor_prev)
dd['mabin']=pd.cut(dd.ma,[0,55,65,75,200]); dd['rbin']=pd.cut(dd.nor_prev,[0,0.15,0.3,0.5,0.7,1.01])
for (mb,rb),s in dd.groupby(['mabin','rbin'],observed=True):
    if len(s)>2000:
        lr = LinearRegression().fit(np.c_[s.ma, s.nor_prev, s.nor], s.ma_p1); print(mb, rb, len(s), 'coef on nor', round(lr.coef_[2],2))
```