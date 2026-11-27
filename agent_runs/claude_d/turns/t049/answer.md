That run failed because lagged MAP has missing values (NaNs). I'll drop those rows and rerun the same NE linearity check.

```python
ok = ~np.isnan(mam1)&~np.isnan(dR)
rt, rtm1, en, ma0, ma1, mam1, dR = [a[ok] for a in (rt, rtm1, en, ma0, ma1, mam1, dR)]
print('n', len(rt))
qs = np.quantile(rtm1,[0,.25,.5,.75,1.0])
for j in range(4):
    s_=(rtm1>=qs[j])&(rtm1<=qs[j+1])
    b,se=ols(ma1[s_], np.column_stack([dR[s_], ma0[s_], mam1[s_], rtm1[s_]]))
    b2,se2=ols(ma1[s_], np.column_stack([en[s_], ma0[s_], mam1[s_], rtm1[s_]]))
    print('prev rate %.2f-%.2f n=%d: per unit rate %.2f(%.2f) | per log-unit %.2f(%.2f)'%(qs[j],qs[j+1],s_.sum(),b[1],se[1],b2[1],se2[1]))
b,se=ols(ma1, np.column_stack([dR, dR**2, dR*(rtm1-0.5), ma0, mam1, rtm1]))
print('pooled dR %.2f(%.2f) dR^2 %.2f(%.2f) dR*(r-.5) %.2f(%.2f)'%(b[1],se[1],b[2],se[2],b[3],se[3]))
```