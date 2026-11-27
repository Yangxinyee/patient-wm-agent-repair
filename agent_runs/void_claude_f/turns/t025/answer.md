Two things are now clear about the norepinephrine policy:
- **Trigger:** in stays without an arterial line, cuff MAP drives the titration with the same thresholds, below 65 up and above 75 down, and the step sizes match those in arterial stays.
- **Rate behaviour:** rates bounce around a lot from hour to hour, are rounded to 0.01 and are capped at 1.0. In the examples, MAP drifts slowly while the rate swings between 0.5 and 1.0, with no obvious response.

The cap is a likely source of bias in my innovation analysis. Patients near 1.0 cannot be stepped up, so their "innovations" are pushed negative, and they are the sicker ones.

Next I will redo the shock analysis with these fixes:
- use the correct trigger reading (arterial in arterial stays, else cuff);
- keep only records with the previous rate between 0.1 and 0.4, so neither the cap nor the 0.01 rounding interferes;
- estimate band means within that subsample;
- use the change in MAP as the outcome, with richer history controls.

If the positive association vanishes, it was a cap artefact and norepinephrine is likely inert. If it persists at a stable size across lags, it is a real effect.

```python
art_map = r.stay_id.map(art_st)
r['M_trig2'] = np.where(art_map, r.map_arterial, r.map_cuff)
r['band2'] = np.where(np.isnan(r.M_trig2), 3, mband(np.nan_to_num(r.M_trig2, nan=70)))
r['Mprev_trig'] = np.where(art_map, ne.loc[r.index,'map_arterial_prev'], ne.loc[r.index,'map_cuff_prev'])
sub = r[(r.ne_prev>=0.1)&(r.ne_prev<=0.4)].copy()
sub['innov2'] = sub.lratio - sub.groupby('band2').lratio.transform('mean')
print(sub.groupby('band2').lratio.agg(['mean','std','count']))
for k in [1,2,3,4,6,8]:
    for nm, base in [('Ma','map_arterial'),('Mc','map_cuff')]:
        dd = sub[sub[f'{nm}_{k}'].notna() & sub[base].notna()]
        dd = dd[(dd.band2<3)]
        y = dd[f'{nm}_{k}'].values - dd[base].values
        X = np.column_stack([dd.innov2, dd[base], dd.M_trig2, dd.lrate_prev, dd.band2==1, dd.band2==2, np.nan_to_num(dd.Mprev_trig, nan=70), np.isnan(dd.Mprev_trig)])
        be,se = ols(y, X)
        print(f"{nm} k={k} n={len(dd)} coef innov2 {be[1]:.3f} +- {se[1]:.3f}  (mean rate {dd.norepinephrine.mean():.3f})")
```