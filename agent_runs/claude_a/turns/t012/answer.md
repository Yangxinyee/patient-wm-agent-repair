Run 11 shows that norepinephrine is titrated:
- It starts at about 0.05 when MAP drops below roughly 63–65, with certainty at 60 or below, and never starts at higher MAP.
- After that, the rate depends mainly on the previous rate and modestly on MAP, with lower MAP giving higher rates.

The rate is therefore a persistent process. The identifying variation has to be the hour-to-hour titration noise, if that is exogenous.

Next I'll model the titration step: the change in rate (difference and log-ratio) given the previous rate and current MAP. Then I'll check whether the residual is iid. Specifically, it should not be autocorrelated and not correlated with the MAP trend or past residuals. Exogenous residuals would let me identify the norepinephrine→MAP effect, as with insulin. Residuals tracking unrecorded state would mean confounding.

```python
c = H[(H.N_prev>0)&(H.N>0)&H.map_now.notna()].copy()
c['dN'] = c.N - c.N_prev
c['lr'] = np.log(c.N/c.N_prev)
print(c.groupby(pd.cut(c.map_now,[0,50,55,60,62,64,65,66,68,70,75,80,90,200])).agg(dN=('dN','mean'), dN_sd=('dN','std'), lr=('lr','mean'), lr_sd=('lr','std'), n=('dN','size')).round(3).to_string())
# stopping
s = H[(H.N_prev>0)&H.map_now.notna()]
print(s.groupby(pd.cut(s.map_now,[0,60,65,70,75,80,85,90,95,200])).agg(p_stop=('N',lambda v:(v==0).mean()), n=('N','size')).round(3).to_string())
print(s.groupby(pd.cut(s.N_prev,[0,0.05,0.1,0.2,0.4,1.0])).agg(p_stop=('N',lambda v:(v==0).mean()), n=('N','size')).round(3).to_string())
```