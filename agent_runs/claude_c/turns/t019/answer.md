Arterial MAP drives norepinephrine decisions when both readings exist; with arterial < 65 the drug starts 100% of the time. The titration rule is on log(rate/previous rate):

| MAP band | Mean change in log rate |
|---|---|
| < 65 | +0.03 to +0.15 (smaller at very low MAP, probably because of the 1.0 cap) |
| 65–75 | about −0.02 to −0.03 (hold) |
| > 75 | about −0.2 (wean) |

The noise sd is about 0.3 in every band. The structure matters for estimation. Because the rate is a titrated random walk, each hour's innovation carries forward into all later rates. A simple regression on the dose at h therefore mixes the effects of many hours of NE. I will need the policy innovations as instruments and then solve a triangular system for the structural per-hour response.

Next I will look at the multiplier distribution in more detail, separately for MAP < 65, 65–75, > 75 and for no MAP reading. That shows whether the rule is "×factor plus lognormal noise, rounded" and lets me define innovations properly.

```python
c['ratio'] = (c.NE/c.NEp)
for lo,hi in [(0,55),(55,65),(65,70),(70,75),(75,200)]:
    x = c[(c.m>=lo)&(c.m<hi)&(c.NEp<1.0)&(c.NEp>=0.2)]
    print(lo,hi,len(x), np.percentile(x.lr,[5,25,50,75,95]).round(3), "mean",x.lr.mean().round(3), "sd", x.lr.std().round(3))
x = c[(c.src=='none')&(c.NEp<1.0)&(c.NEp>=0.2)]
print('none',len(x), np.percentile(x.lr,[5,25,50,75,95]).round(3), x.lr.mean().round(3), x.lr.std().round(3))
# is the MAP<65 vs >=65 rule sharp at 65 with arterial?
ca = c[(c.src=='a')&(c.NEp<0.8)&(c.NEp>=0.2)]
print(ca.groupby(ca.m.round()).lr.mean().loc[55:85].round(3).to_dict())
# stop prob vs MAP for continuing
q2 = q[q.NEp.notna()].copy(); q2['m']=q2.map_a.fillna(q2.map_c)
print(q2.groupby(q2.m.round().clip(50,95)).NE.apply(lambda x: x.isna().mean()).loc[60:95].round(3).to_dict())
```