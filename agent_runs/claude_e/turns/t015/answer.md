Norepinephrine is titrated. It starts once MAP falls below about 65 and, once started, almost always continues (97.5%). Each hour the rate is adjusted from the previous one:
- up about 4–9% when MAP is below 65,
- unchanged at 65–75,
- down about 15% above 75.

The spread around these steps is sizeable (SD of the change about 0.12–0.18 mcg/kg/min). Because the rate accumulates its own history, its level reflects how sick the patient has been. A level comparison would be confounded, so the usable variation is the hour-to-hour change that is random given the recorded history.

First I need the exact titration rule:
- multiplicative or additive steps;
- which MAP it uses (arterial at this hour, cuff, or the latest);
- what the random part looks like.

The test: if the rule depends only on recorded values, the residual change should be unrelated to anything else recorded.

```python
C['mapnow'] = C.ma_now
C['mc_now'] = Gtr['mc'].stack(dropna=False).reindex(pd.MultiIndex.from_arrays([C.stay,C.h])).values
C['lr'] = np.log(C['ne']/C.ne_prev)
for lo,hi in [(40,50),(55,60),(60,65),(65,70),(70,75),(75,80),(80,90)]:
    s = C[(C.mapnow>=lo)&(C.mapnow<hi)]
    print(lo,hi,len(s),'log-ratio quantiles', np.round(s.lr.quantile([.05,.25,.5,.75,.95]).values,3), ' diff q', np.round(s.dr.quantile([.05,.25,.5,.75,.95]).values,3))
s = C[(C.mapnow>=65)&(C.mapnow<75)&(C.ne_prev<0.95)]
print('example ratios (65-75):', s.ratio.round(3).value_counts().head(15).to_string())
```