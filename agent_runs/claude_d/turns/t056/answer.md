These estimates are internally inconsistent. Low- and high-rate instruments give lag-1 effects of 0.3 vs 2.5 per log-unit, about 5 sd apart. Yet the two groups cover a narrow range of rates (about 0.16–0.3 vs 0.3–0.5). The heterogeneity runs the same way under every coding, which also contradicts run 46.

So I suspect a problem with the instrument or with my IV function, not real dose-response curvature. I'll run three simple checks:
1. Is the new instrument Zc predictable from pre-treatment history (MAP at t, t−1, t−2, previous rates)? If so, it is not clean.
2. Within the interior stratum, what are the plain lag-1 reduced form (MA_{t+1} on Zc_t, controlling for MA_t) and first stage, for low vs high previous rate?
3. Do these match the IV γ₁?

```python
rows=[]
for t in range(2,24):
    s_ = Sst[:,t]&~np.isnan(MA[:,t+1])&~np.isnan(MA[:,t])&~np.isnan(MA[:,t-1])
    idx=np.where(s_)[0]
    rows.append(np.column_stack([Zc[idx,t], MA[idx,t+1], MA[idx,t], MA[idx,t-1], NOR[idx,t-1], NOR[idx,t-2], codings['log'][idx,t], NOR[idx,t], np.full(len(idx),t)]))
Q=np.vstack(rows); zq,y1q,y0q,ym1q,rp1,rp2,alq,rq,tq = Q.T
b,se = ols(zq, np.column_stack([y0q,ym1q,rp1,rp2,tq]))
print('Z on history: coefs', b[1:].round(4), 'R2-ish', np.corrcoef(zq, np.column_stack([np.ones(len(zq)),y0q,ym1q,rp1,rp2,tq])@b)[0,1]**2)
for nm_, s_ in [('low prev<.3', rp1<0.3), ('high prev>=.3', rp1>=0.3), ('all', np.ones(len(zq),bool))]:
    rf = ols(y1q[s_], np.column_stack([zq[s_], y0q[s_], ym1q[s_]]))
    fs = ols(alq[s_], np.column_stack([zq[s_], y0q[s_], ym1q[s_]]))
    print('%-14s n=%6d RF %.2f(%.2f) FSlog %.3f -> lag1 per log %.2f; mean MAP_t %.1f'%(nm_, s_.sum(), rf[0][1], rf[1][1], fs[0][1], rf[0][1]/fs[0][1], y0q[s_].mean()))
```