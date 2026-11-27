The insulin policy has a clear structure:
- The rate is essentially 0 below 140 mg/dL, sometimes on between 140 and 160 (36%), and always on above 160.
- The mean rate climbs in steps: about 1.1 at 160–180, 2.1 at 180–220, 3.1 at 220–250, and 6.3 above 300.
- Within a glucose band the rate still varies a lot (SD 0.4–2.0).
- A fresh finger-stick is always present when insulin continues, so glucose is checked hourly on the drip.

That within-band variation matters most. It could be exogenous noise, which would give a natural experiment for the causal effect, or it could reflect something clinicians saw that the record doesn't (hidden glucose, trend), which would be confounding.

Next I'll compare finger and lab glucose taken in the same hour, to learn the measurement structure: bias, noise, and whether one tracks a latent "true" glucose. I'll also check how reading frequency depends on treatment status. Both shape what "true change in glucose_finger" means and what the dose can be confounded with.

```python
both = d[d.glucose_finger.notna() & d.glucose_lab.notna()]
print("n both", len(both))
print((both.glucose_finger - both.glucose_lab).describe())
print(np.corrcoef(both.glucose_finger, both.glucose_lab))
# regress finger on lab
from numpy.polynomial import polynomial as Pn
b = np.polyfit(both.glucose_lab, both.glucose_finger, 1); print("finger ~ lab:", b)
b2 = np.polyfit(both.glucose_finger, both.glucose_lab, 1); print("lab ~ finger:", b2)
# MAP cuff vs arterial
bm = d[d.map_arterial.notna() & d.map_cuff.notna()]
print("n both map", len(bm)); print((bm.map_cuff-bm.map_arterial).describe()); print(np.corrcoef(bm.map_cuff,bm.map_arterial))
# finger measurement frequency by insulin status
d['on_ins'] = d.insulin>0
print(d.groupby('on_ins').glucose_finger.apply(lambda x: x.notna().mean()))
print(d.groupby(st.has_ins.reindex(d.stay_id).values).glucose_finger.apply(lambda x: x.notna().mean()))
```