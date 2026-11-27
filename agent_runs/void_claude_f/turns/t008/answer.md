Insulin is given only when finger glucose is above 140. The rate rises steeply with glucose: about 1 U/h at 140–180 mg/dL, 2 at 180–220, 3 at 220–260 and 4 at 260–300. Within each band the spread is real (sd 0.4–2.0), and 98% of insulin records follow a finger reading in the same hour. So the clinician mostly doses off the current finger stick, with a sizeable leftover component.

I will now check:
- whether that leftover is noise from hour to hour or a persistent per-stay trait, by regressing the rate on current and past readings and looking at residual correlations within stays;
- the probability of starting insulin at each glucose level.

This decides the identification strategy. Independent hourly noise in the dose would act like a randomised experiment. A per-stay dosing scale could be a clinician effect (usable) or a patient insulin-resistance effect (confounded).

```python
x = PN.reset_index().sort_values(['stay_id','h'])
# previous values within stay
for c in ['glucose_finger','glucose_lab','insulin','norepinephrine','map_arterial','map_cuff']:
    x[c+'_prev'] = x.groupby('stay_id')[c].shift(1)
# P(insulin | finger glucose bin)
xf = x[x.glucose_finger.notna() & (x.h>=0)].copy()
xf['gb'] = pd.cut(xf.glucose_finger, [0,100,120,130,135,140,145,150,160,180,220,260,300,500])
print(xf.groupby('gb').insulin.apply(lambda s: s.notna().mean()).to_string())
# when finger glucose missing, is insulin given?
print("P(ins | no finger glucose):", x[x.glucose_finger.isna() & (x.h>=0)].insulin.notna().mean())
# examine rate vs G more precisely in insulin rows
mi = x[x.insulin.notna() & x.glucose_finger.notna()].copy()
import numpy.linalg as la
mi['Gc'] = mi.glucose_finger
for g0,g1 in [(140,150),(150,160),(190,200),(240,250),(290,300)]:
    s = mi[(mi.Gc>=g0)&(mi.Gc<g1)].insulin
    print(g0,g1, s.value_counts().sort_index().to_dict())
```