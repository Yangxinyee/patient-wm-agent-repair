Two things stand out.

1. About 9% of the "insulin" bundles have zero recorded rate. These are probably bundles holding only the END atom of the previous hour's infusion, so I'll drop them.
2. Neither z nor the full 992-dim state linearly encodes the dose surprise (R² ≈ 0). The raw rate is only partly decodable (R² 0.66), about what the policy base level alone explains (corr(rate, base) = 0.79). So after an insulin bundle the model's state keeps mostly "what dose the glucose implies", not the dose actually given. That fits its weak response.

I'll repeat the analysis on true dosing bundles only, and check how EPS is defined. I'll also compare out, h slots and z separately, and look at what the hh:10 (0.17) bundles are.

```python
ok=rate>0; print('n ins bundles',ok.sum())
r2=lambda XX,y_: cross_val_score(RidgeCV(alphas=np.logspace(-3,4,15)),XX,y_,cv=5,scoring='r2').mean().round(3)
y_s=rate-1.04*base
for nm_,y_ in [('eps',eps[ok]),('rate-1.04b',y_s[ok]),('rate',rate[ok]),('base',base[ok])]:
    print(nm_, 'z',r2(Z[ok],y_), 'out',r2(X3[iu][ok,:192],y_), 'h',r2(X3[iu][ok,192:960],y_), 'full',r2(X3[iu][ok],y_))
lr_=LinearRegression().fit(np.c_[rate[ok],base[ok]],eps[ok]); print('eps~rate+base coef',lr_.coef_.round(3),lr_.intercept_.round(3))
f17=np.isclose(hr3%1,0.17,atol=0.01); print('n frac .17',f17.sum(),'ins',hi3[f17].mean().round(2),'ne',hn3[f17].mean().round(2),'atoms',np.bincount(S3['n_atoms'][f17].astype(int))[:6])
```