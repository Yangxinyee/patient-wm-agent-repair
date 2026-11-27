Now norepinephrine shows a signal. A positive titration shock raises later MAP: arterial +0.33 ± 0.05 per unit of log-rate shock at 1 hour, fading by 4–8 hours; cuff +0.76 ± 0.11, persisting to 12 hours.

The effect is small in natural units. With a typical rate around 0.5–0.6, it is roughly 0.5–1.5 mmHg per mcg/kg/min. That would make a doubling worth well under 1 mmHg.

I do not trust it fully yet, for three reasons:
- the earlier rate-in-levels regression showed zero for arterial MAP;
- arterial and cuff should shift equally if the drug moves true MAP;
- my "innovation" also absorbs effects of the 1.0 cap and the 0.01 rounding floor, and those relate to how sick the patient is.

Next I will use a cleaner, large-contrast design: the start threshold at arterial MAP 65.
- Compare MAP 1–6 hours later for stays just below 65 (norepinephrine starts) vs just above (usually no start).
- Scale the jump by the jump in rate (a fuzzy threshold comparison).
- Restrict to hours with no norepinephrine in the previous hour.

A clear positive MAP jump would confirm a real vasopressor effect and give its size per unit rate. Near-zero would point to an artefact in the innovation analysis.

```python
ne['prev_on'] = ne.ne_prev.notna()
rdn = ne[(~ne.prev_on) & ne.map_arterial.between(55,75) & (ne.h>=0) & (ne.h<=20)].copy()
rdn['below'] = (rdn.map_arterial<=64).astype(float)   # 65 is the ambiguous value
rdn = rdn[rdn.map_arterial!=65]
rdn['mc'] = rdn.map_arterial - 65
rdn['ne0'] = rdn.norepinephrine.fillna(0)
print("first stage: P(start) below/above", rdn.groupby('below').norepinephrine.apply(lambda s: s.notna().mean()).to_dict(),
      "mean rate", rdn.groupby('below').ne0.mean().to_dict())
for k in [1,2,3,4,6]:
    for nm in ['Ma','Mc']:
        d = rdn[rdn[f'{nm}_{k}'].notna()]
        X = np.column_stack([d.below, d.mc, d.mc*d.below])
        be,se = ols(d[f'{nm}_{k}'].values, X)
        be1,se1 = ols(d.ne0.values, X)
        print(f"RD {nm} k={k}: MAP jump {be[1]:.2f}+-{se[1]:.2f}, rate jump {be1[1]:.3f}, ratio {be[1]/be1[1]:.2f} mmHg per mcg/kg/min, n={len(d)}")
```