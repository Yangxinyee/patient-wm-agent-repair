The insulin multiplier is essentially uncorrelated with everything: |r| ≤ 0.015 with glucose, lab glucose, MAP, norepinephrine, hour and previous dose. The small residual correlations are probably artefacts of rounding to 0.5 within strata, and I will stratify by glucose anyway. When no glucose_finger reading exists at h, the scale is applied to the glucose_lab value instead (lab 154 gives about 1 u/h, lab 182 gives 1.5). So insulin dosing is a sequentially randomised design given the latest glucose reading.

Next is the norepinephrine policy. I will check:
- the rate against MAP at h (arterial when present, else cuff);
- whether it depends on the previous rate (titration);
- the cap at 1.0;
- whether there is a random multiplier with no autocorrelation.

The same natural experiment for norepinephrine would confirm that both effects can be identified by randomisation.

```python
ne = Q[Q.norepinephrine.notna()].copy()
ne['map_obs'] = ne.map_arterial.fillna(ne.map_cuff)
ne['src'] = np.where(ne.map_arterial.notna(),'art', np.where(ne.map_cuff.notna(),'cuff','none'))
print(ne.src.value_counts())
ne['mbin'] = (ne.map_obs//5*5)
print(ne.groupby(['src','mbin']).norepinephrine.agg(['size','mean','std','min','max']).round(3).to_string())
# P(ne) by MAP bins (where MAP observed)
Q['map_obs'] = Q.map_arterial.fillna(Q.map_cuff)
qq = Q[Q.map_obs.notna()].copy(); qq['mbin']=(qq.map_obs//5*5)
qq['src'] = np.where(qq.map_arterial.notna(),'art','cuff')
print(qq.groupby(['src','mbin']).norepinephrine.apply(lambda x: x.notna().mean()).round(3).unstack(0).to_string())
```