The base rate is a step function of the policy glucose: 1 at 150–180, 2 at 180–220, 3 at 220–260, 4 at 260–300 and 6 above 300. On top of it sits a multiplicative jitter with mean about 1.05× base and standard deviation about 0.33× base, rounded to 0.5 units.

Now the key question is whether the jitter is exogenous. I will compute ratio = rate/base and check:
- its autocorrelation across consecutive hours in a stay;
- how much of its variance is a stay-level component;
- its correlation with the glucose trend, with the position within the band, and with a simultaneous lab glucose given the finger reading.

If all are near zero, apart from the mechanical effect of rounding, the jitter behaves like a randomized dose perturbation. A strong stay-level component would suggest clinicians tailor doses to patient traits, which would be a confounder.

```python
def base_rate(g):
    return np.select([g<=150, g<=180, g<=220, g<=260, g<=300], [0,1,2,3,4], 6)
q2['base'] = base_rate(q2.pol.values)
q2 = q2.sort_values(['stay_id','h'])
on = q2[(q2.pol>150)&(q2.insulin>0)].copy()
on['ratio'] = on.insulin/on.base
on['lr'] = np.log(on.ratio)
print("ratio overall:", on.ratio.describe().round(3).to_dict())
# lag-1 autocorrelation (consecutive hours, same stay)
on['lr_prev'] = on.groupby('stay_id').lr.shift(1); on['h_prev'] = on.groupby('stay_id').h.shift(1)
c = on[(on.h-on.h_prev)==1]
print("lag1 corr log ratio:", np.corrcoef(c.lr, c.lr_prev)[0,1].round(4), len(c))
# within same band only (avoid rounding artifacts across bands)
cb = c[c.base==c.groupby('stay_id').base.shift(1).reindex(c.index)]
# stay-level variance component (stays with >=10 on-hours)
cnt = on.groupby('stay_id').lr.transform('size')
big = on[cnt>=10]
sm = big.groupby('stay_id').lr.mean()
print("var(lr):", big.lr.var().round(4), " var of stay means:", sm.var().round(4), " expected if iid:", (big.lr.var()/big.groupby('stay_id').size().mean()).round(4))
# trend
on['pol_prev'] = on.groupby('stay_id').pol.shift(1)
t = on[(on.h-on.h_prev)==1]
print("corr lr with trend:", np.corrcoef(t.lr, t.pol - t.pol_prev)[0,1].round(4))
# within band position
for b in [1,2,3]:
    sb = on[on.base==b]
    print("band",b," corr lr with pol:", np.corrcoef(sb.lr, sb.pol)[0,1].round(4), " n", len(sb))
# simultaneous lab given finger
sim = on[(on.glucose_finger_age==0)&(on.glucose_lab_age==0)].copy()
import statsmodels.api as sm_ if False else None
from numpy.linalg import lstsq
X = np.column_stack([np.ones(len(sim)), sim.pol, sim.base, sim.glucose_lab])
coef = lstsq(X, sim.lr.values, rcond=None)[0]
print("regress lr on [1,pol,base,lab]:", coef.round(5), " n", len(sim))
print("corr(lr, lab - finger):", np.corrcoef(sim.lr, sim.glucose_lab - sim.pol)[0,1].round(4))
```