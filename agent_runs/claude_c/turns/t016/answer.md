Several things are now clear.
- **Skips:** every skipped dose is at hour −1, the pre-ICU reading. In the ICU, insulin is always given when the reading is ≥ 150. The only random element is the rate noise, which the placebo tests support as exogenous.
- **Effect size:** about −1.9 to −2.3 mg/dL per U/h at 1 h and 2 h, in every tier. It does not depend on glucose level.
- **Dose-response:** within tiers 2 and 3 the residual next-hour glucose falls roughly linearly with dose, at about −2 per U/h, from 1 to about 6 U/h. There is no visible saturation, so linear extrapolation to doubled doses is plausible.

Next I will write these findings to the notebook. Then I will estimate the full insulin response curve for lags 1–12 from given doses only, adding controls for earlier glucose to tighten the estimates. That curve, applied to each stay's recorded doses from hour 12, gives the predicted true ×2 and ×½ readouts. A smooth decay to about 0 by 8–10 h would confirm a transient, linear effect.

```python
notes = """# Notebook (ICU world-model dose repair)

## Data facts (runs 1-15)
- All stays 25.17 h; readings at integer hours (-1..24), drug records at h+0.0833 lasting 1 h.
- Insulin policy (deterministic given reading): uses finger glucose at hour h (else lab if no finger). No insulin if <150;
  tiers base rate 1/2/3/4(~4.2) U/h at thresholds 150/180/220/260 (50/50 at exact threshold values).
  Rate = base * lognormal noise (log-sd ~0.3) rounded to 0.5. Noise is iid (lag1 corr 0.004, no stay effect,
  no corr with pre-treatment vars). ALL 'skips' are at hour -1 (pre-ICU) -> skip variation is confounded, don't use.
- NE: capped at 1.0 mcg/kg/min (13.6% at cap); given ~always when MAP<65; analysis pending.
## Insulin causal effect (from within-tier rate noise among given doses, finger-tier hours)
- dG(finger) per U/h of a 1-h dose: L=1 -1.98, L=2 -2.23, L=3 -1.75, L=4 -1.27, L=6 -0.45, L=8 -0.14, L=10/12 ~0.
- Same across tiers (no glucose-level dependence); roughly linear in dose (no saturation seen up to ~6 U/h).
- Naive regressions including skips give a fake 'rebound' (+2 at 12h) -> artifact.
"""
open('/work/notes.md','w').write(notes)

def irf_insulin(P, Ls, extra_controls=True):
    out=[]
    Pg = P.groupby(level=0)
    prevs = {k: Pg.glucose_finger.shift(k) for k in [1,2,3]}
    for L in Ls:
        GL = Pg.glucose_finger.shift(-L)
        q = P.assign(GL=GL, gp1=prevs[1], gp2=prevs[2], gp3=prevs[3])
        q = q[q.glucose_finger.notna() & (q.tf>=1) & q.insulin.notna() & q.GL.notna()]
        q = q[~q.glucose_finger.round().isin([150,180,220,260])]
        g = q.glucose_finger.values
        X = [np.ones(len(q)), g, g**2/100, g**3/1e4] + [(q.tf.values==t).astype(float) for t in [2,3,4]]
        nm = ['c','g','g2','g3','t2','t3','t4']
        if extra_controls:
            for k in [1,2,3]:
                v = q[f'gp{k}'].values; has = ~np.isnan(v)
                X += [np.where(has, v, 0.0), has.astype(float)]; nm += [f'gp{k}', f'has{k}']
            hh = q.index.get_level_values('h').values
            X += [hh/24.0]; nm += ['hh']
        X += [q.D.values]; nm += ['D']
        r = ols(q.GL.values, X, nm).loc['D']
        out.append((L, len(q), r.b, r.se))
    return pd.DataFrame(out, columns=['L','n','b','se'])
irf = irf_insulin(P, range(1,13))
print(irf.round(3))
```