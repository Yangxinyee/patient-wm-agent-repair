Insulin is clearly linear and additive. The lag-1 effect is about −1.9 to −2.0 mg/dL per u/h at every base level (glucose 164 to 272), with only a slight quadratic hint (−0.08 ± 0.04). So the linear impulse response beta_ins is the right truth model, and the ×2 and ×½ effects should stand about −2:1.

Now the same check for NE. Within titrated hours that are not capped, the random part of the rate change comes from the log-noise EPSN. I'll compare the MAP response at lag 1 per unit of rate change against per unit of log-rate change, across rate levels. If the per-unit-rate effect stays constant across levels, NE is linear in rate. If the per-log-unit effect stays constant instead, it is log-linear, and ×2 and ×½ would then be symmetric.

```python
rows=[]
for t in range(1,23):
    ok = (NOR[:,t]>0)&(NOR[:,t-1]>0)&(NOR[:,t]<0.97)&~np.isnan(MA[:,t+1])&~np.isnan(MA[:,t])&(EPSN[:,t]!=0)
    idx=np.where(ok)[0]
    rows.append(np.column_stack([NOR[idx,t], NOR[idx,t-1], EPSN[idx,t], MA[idx,t], MA[idx,t+1], MA[idx,t-1] if t>=1 else MA[idx,t]]))
Rn=np.vstack(rows); rt, rtm1, en, ma0, ma1, mam1 = Rn.T
dR = rt - rt*np.exp(-en)   # rate change due to noise
print('n',len(Rn), 'rate quantiles', np.quantile(rt,[.1,.5,.9]).round(2))
qs = np.quantile(rtm1,[0,.25,.5,.75,1.0])
for j in range(4):
    s_=(rtm1>=qs[j])&(rtm1<=qs[j+1])
    b,se=ols(ma1[s_], np.column_stack([dR[s_], ma0[s_], mam1[s_], rtm1[s_]]))
    b2,se2=ols(ma1[s_], np.column_stack([en[s_], ma0[s_], mam1[s_], rtm1[s_]]))
    print('prev rate %.2f-%.2f n=%d: per unit rate %.2f(%.2f) | per log-unit %.2f(%.2f)'%(qs[j],qs[j+1],s_.sum(),b[1],se[1],b2[1],se2[1]))
b,se=ols(ma1, np.column_stack([dR, dR**2, dR*(rtm1-0.5), ma0, mam1, rtm1]))
print('pooled dR %.2f(%.2f) dR^2 %.2f(%.2f) dR*(r-.5) %.2f(%.2f)'%(b[1],se[1],b[2],se[2],b[3],se[3]))
```