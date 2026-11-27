Each drug always comes 5 minutes after the hour's readings. The populations are distinct:
- about half of stays have an arterial line (map_arterial nearly every hour) and half never do;
- about 38% of stays never get insulin, while 29% get it in more than 80% of hours;
- about 56% never get norepinephrine, while 28% get it in more than 80% of hours.

Next I need the dosing policy: how the insulin rate at hour t+5min depends on the hour-t readings and the previous rate. If the rate is a noisy function of observed readings, adjusting for those readings identifies the effect. If it is a step function (a sliding scale), the thresholds give a regression-discontinuity design. If it depends on something unobserved, I need another strategy. I will tabulate the insulin rate against the same-hour glucose_finger, and the norepinephrine rate against the same-hour MAP, including the previous rate.

```python
P = P.copy()
P['ins'] = P.insulin.fillna(0); P['ne'] = P.norepinephrine.fillna(0)
g = P.groupby(level=0)
P['ins_prev'] = g.ins.shift(1); P['ne_prev'] = g.ne.shift(1)
P['gf'] = P.glucose_finger; P['gl']=P.glucose_lab; P['ma']=P.map_arterial; P['mc']=P.map_cuff
# insulin vs glucose finger same hour
d = P[P.gf.notna()]
d['gbin'] = pd.cut(d.gf, [0,100,120,140,150,160,170,180,190,200,220,240,260,300,500])
print(d.groupby('gbin').ins.agg(['mean','std','size', lambda x:(x>0).mean()]).to_string())
print(pd.crosstab(d.gbin, d.ins.clip(upper=8)).to_string())
```