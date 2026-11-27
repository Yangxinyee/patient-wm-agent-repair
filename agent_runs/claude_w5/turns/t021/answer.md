The norepinephrine noise is multiplicative: log(rate_t / rate_{t−1}) has sd about 0.30 at every rate level, until the 1.0 cap truncates it. The MAP-driven step looks additive in absolute rate units, constant across rate levels below 0.6:

| MAP | step |
|---|---|
| < 55 | about +0.08 to +0.10 |
| 55–65 | +0.04 to +0.05 |
| 65–75 | about 0 |
| > 75 | about −0.035 |

A candidate policy is rate_t = clip((rate_{t−1} + Δ(MAP)) × exp(η), 0.01, 1.0), with η sd about 0.3.

Next I'll pin down Δ(MAP) finely. Under the candidate policy, the median of rate_t − rate_{t−1} equals Δ, because median exp(η) = 1. I'll compute it for previous rates of 0.2–0.6 (away from floor and cap) in 1-mmHg MAP bins, separately for arterial and cuff, and for hours with no MAP reading. Sharp steps at fixed thresholds would show the policy uses the recorded MAP, like insulin.

```python
mid = ne[(ne.norepinephrine_prev>=0.2)&(ne.norepinephrine_prev<=0.6)].copy()
for src in ['art','cuff']:
    d = mid[mid.map_src==src]
    mb = (d.map_now//2)*2
    med = d.groupby(mb).dr.median()
    nn = d.groupby(mb).size()
    out = pd.DataFrame({'n':nn, 'med_dr':med}).loc[40:96]
    print(src, out.T.round(3).to_string())
d = mid[mid.map_src=='none']; print('none median dr', d.dr.median(), len(d))
# check if arterial preferred when both present
both = mid[mid.map_arterial.notna() & mid.map_cuff.notna()]
for col in ['map_arterial','map_cuff']:
    print(col, both.groupby(pd.cut(both[col],[0,55,60,65,70,75,80,200])).dr.median().round(3).to_dict())
```