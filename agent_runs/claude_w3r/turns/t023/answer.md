The norepinephrine reduced form is clearly positive: about +2.3 mmHg MAP at 1 h per unit rate innovation, peaking near +4.5 at 3 h, carried by the persistent rate. The deconvolved structural response is +2.34, +2.41, +1.30, +0.79 at lags 1–4 and about 0 after, a cumulative effect of roughly 7 mmHg per (mcg/kg/min·h).

The placebo fails, though. The innovation is correlated with the previous MAP (−1.73 ± 0.10). The titration therefore also depends on MAP history (trend), or on something that MAP history reveals, so my conditioning set is incomplete. Next I will regress the rate on several candidate policy inputs (MAP lags, cuff MAP, previous rates) to find what the titration really uses, then refit the innovation.

```python
art2 = art.copy()
t_idx = art2.t.values; s_idx = art2.stay.values
art2['m1'] = ma_full[s_idx, t_idx-1]; art2['m2l'] = ma_full[s_idx, np.maximum(t_idx-2,0)]
art2['mc0'] = mc[s_idx, t_idx]; art2['rp2'] = ne_r[s_idx, np.maximum(t_idx-2,0)]
art2['g0'] = gf_last[s_idx, t_idx]
o = art2[(art2.rprev>0)&(art2.r>0)&(art2.r<0.999)].dropna(subset=['m1','m2l'])
o['dr']=o.r-o.rprev
for X_ in [['m'],['m','m1'],['m','m1','m2l'],['m','m1','m2l','rprev','rp2'],['m','m1','mc0']]:
    oo=o.dropna(subset=X_); b,s_,nn=ols(oo,'dr',X_)
    resid = oo.dr - (b.const + sum(b[c]*oo[c] for c in X_))
    print(X_, 'coefs', b.round(4).to_dict(), 'R2 %.3f'%(1-resid.var()/oo.dr.var()), 'n', nn)
```