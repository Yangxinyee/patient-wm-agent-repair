The same pipeline applied to insulin gives an implied x2 readout of −0.04 (bootstrap mean −0.17, SD 0.49), consistent with zero. So the method does not invent effects, but it also says the insulin truth is only pinned to within about ±1 mg/dL.

For norepinephrine the evidence for a small negative effect (pooled −0.32 ± 0.14 for x2) remains moderate. Its pattern (near zero at 1 hour, building over later hours) would fit a delayed drug effect.

The analyses so far use only the titration noise of continuing infusions. Stopping decisions may be a stronger source of variation: above MAP 75, about 23% of hours stop norepinephrine. A stop removes the whole rate (about 0.2–0.4) for the following hours, a much bigger dose contrast than an innovation of SD 0.14. Next I will check whether stops are random given observables:
- P(stop) against exact MAP and the previous rate;
- whether stop decisions correlate with the prior MAP trend or the cuff reading.

If stops are as-good-as-random, a stop-based estimate can sharpen the norepinephrine effect.

```python
st_ = P[(P['nor_prev']>0) & P['ma'].notna()].copy().reset_index()
st_['stop'] = (st_['nor']==0).astype(float)
print(st_.groupby(st_['ma'].clip(60,95).round())['stop'].agg(['mean','size']).loc[70:90].round(3).T.to_string())
hi = st_[st_.ma>=76].copy()
print("stop by prev-rate bin (MAP>=76):"); print(hi.groupby(pd.cut(hi.nor_prev,[0,0.05,0.1,0.2,0.3,0.5,1.0]))['stop'].agg(['mean','size']).round(3).to_string())
hi['dMprev'] = hi['ma'] - hi['ma_prev']
print("stop vs prior trend:"); print(hi.groupby(pd.cut(hi.dMprev,[-50,-10,-3,0,3,10,50]))['stop'].agg(['mean','size']).round(3).to_string())
print("stop by hour:", hi.groupby(pd.cut(hi.ih,[-1,6,12,18,24]))['stop'].mean().round(3).to_dict())
```