Run 7 shows that insulin follows a near-deterministic sliding scale on the current finger glucose:
- No insulin below 150; insulin always given at 150 and above.
- Rate tiers average about 1.05 for 150–180, 2.1 for 180–220, 3.1 for 220–250 and 3.9 above 250, with spread inside each tier.
- When no finger reading exists, insulin is almost never given (2.5%).

Norepinephrine is always given at MAP ≤ 60 and its probability falls with MAP. Its rate varies a lot and is U-shaped in MAP, which suggests titration with persistence.

So the insulin on/off decision is fully explained by a recorded reading. That leaves two sources of identifying variation: the threshold discontinuity at 150, and the spread of rates within tiers.

Next I'll find out what drives the within-tier rate spread. I'll check its distribution, its correlation within a stay (a clinician/patient factor?), and its relation to previous rate, glucose trend, lab glucose and hour. Exogenous spread would identify the dose-response. Spread tied to an unrecorded state would be confounding.

```python
ins = x[x.ins_on==1].copy()
ins['tier'] = pd.cut(ins.glucose_finger, [0,149.5,179.5,219.5,249.5,1000], labels=['<150','150-179','180-219','220-249','250+'])
print(pd.crosstab(ins.tier, ins.insulin).iloc[:, :14].to_string())
# within tier, by exact glucose
t1 = ins[ins.tier=='150-179']
print(t1.groupby(pd.cut(t1.glucose_finger,[149,155,160,165,170,175,180])).insulin.agg(['mean','std','size']))
# multiplier relative to base tier rate
base = {'150-179':1.0,'180-219':2.0,'220-249':3.0}
ins['mult'] = ins.insulin / ins.tier.map(base).astype(float)
print(ins.groupby('tier').mult.describe())
```