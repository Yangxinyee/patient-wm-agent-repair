Insulin's effect persists. Per 1 u/h given in one hour, the reading k hours later changes by about:

| lag (h) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| change (mg/dL) | −1.99 | −2.27 | −1.75 | −1.29 | −0.70 | −0.47 | −0.22 | −0.15 |

The SEs are about 0.05–0.09. The effect peaks at lag 2, which fits the infusion running until t+1.08, after the t+1 reading. It then decays with a factor of about 0.7 per hour. Summed over lags, a sustained extra 1 u/h lowers glucose by about 9 mg/dL. So doubling from hour 12 should build to a sizeable fall, likely around −10 to −20 mg/dL averaged over hours 13–24.

Two small biases push these "total" numbers toward zero:
- lower glucose triggers lower later doses through the protocol;
- higher doses make later readings slightly more often missing (−0.005 per u/h).

I'll correct for both later with a structural fit.

Next I'll examine the norepinephrine protocol in the same way: how its rate relates to MAP, both arterial and cuff, and to the previous rate (titration). That will show what exogenous variation exists for the MAP effect.

```python
PN = P.copy()
PN['on_ne'] = PN['ne'].notna()
print('P(on NE | prev on NE):', PN.groupby(PN.ne_prev.notna()).on_ne.mean().to_string())
Pm = PN[PN.ma_now.notna()]
print(Pm.groupby(pd.cut(Pm.ma_now,[0,50,55,60,62,64,65,66,68,70,75,80,90,200]),observed=True).agg(n=('ne','size'),p=('on_ne','mean'),r=('ne','mean')).to_string())
# among continuing NE: rate change vs MAP
C = PN[PN.on_ne & PN.ne_prev.notna()].copy(); C['dr'] = C['ne']-C.ne_prev; C['ratio']=C['ne']/C.ne_prev
print('continuing NE, has ma_now:', C.ma_now.notna().mean(), ' has mc_last:', C.mc_last.notna().mean())
print(C.groupby(pd.cut(C.ma_last,[0,50,55,60,63,65,67,70,75,80,90,200]),observed=True).agg(n=('dr','size'),dr=('dr','mean'),ratio=('ratio','median'),sd=('dr','std')).to_string())
```