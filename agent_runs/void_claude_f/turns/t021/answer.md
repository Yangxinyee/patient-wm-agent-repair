When both readings exist, arterial MAP drives the titration: band fit R² is 0.052 for arterial vs 0.015 for cuff. The jump in the rate's drift is at arterial MAP 65; along cuff MAP there is no jump. So my policy model is right in its main structure. The small cuff correlation is probably detail, such as the cap at 1.0 or the size of the step varying with how low MAP is.

Next is the decisive test for norepinephrine's effect: a regression of future MAP (arterial and cuff, lags 1–12 h) on the titration innovation, controlling for current MAP, the previous rate and the band.
- A shock to the log rate persists into all later rates, so any real effect, immediate or delayed, would show up as non-zero coefficients.
- If norepinephrine raises MAP, I expect positive coefficients; zero at all lags would mean it is inert like insulin.

To get the effect in natural units, I will also scale the result per unit of rate change.

```python
r['lrate'] = np.log(r.norepinephrine)
r['lrate_prev'] = np.log(r.ne_prev)
for k in [1,2,3,4,6,8,12]:
    r[f'Ma_{k}'] = r.groupby('stay_id').map_arterial.shift(-k)
    r[f'Mc_{k}'] = r.groupby('stay_id').map_cuff.shift(-k)
# Note: shift within r (only NE-on rows) is wrong for lags; recompute from full panel
ne = ne.sort_values(['stay_id','h'])
for k in [1,2,3,4,6,8,12]:
    ne[f'Ma_{k}'] = ne.groupby('stay_id').map_arterial.shift(-k)
    ne[f'Mc_{k}'] = ne.groupby('stay_id').map_cuff.shift(-k)
cols = [f'Ma_{k}' for k in [1,2,3,4,6,8,12]]+[f'Mc_{k}' for k in [1,2,3,4,6,8,12]]
r = r.drop(columns=[c for c in r.columns if c in cols]).join(ne[cols], how='left')
for k in [1,2,3,4,6,8,12]:
    for nm, base in [('Ma','map_arterial'),('Mc','map_cuff')]:
        dd = r[r[f'{nm}_{k}'].notna() & r[base].notna()]
        X = np.column_stack([dd.innov, dd[base], dd[base]**2/100, dd.lrate_prev, dd.band==1, dd.band==2, dd.band==3])
        be,se = ols(dd[f'{nm}_{k}'].values, X)
        # scale: effect of innovation in units of rate: d rate = rate*innov approx; report per +0.1 log-ratio
        print(f"{nm} k={k} n={len(dd)} coef innov {be[1]:.3f} +- {se[1]:.3f}   mean rate {dd.norepinephrine.mean():.3f}")
```