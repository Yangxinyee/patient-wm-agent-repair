## Step 19: the full 12-hour insulin response, and checking linearity

**What the last result showed.**
- With 12 lags (lab outcome), the response is about −1.6, −1.8, −1.8, −1.4, −1.4, −0.9, −0.5, −0.7, then near 0 (SE about 0.3). The cumulative effect is about −11 mg/dL per sustained unit/h by 12 hours.
- The finger version (8 lags) is very similar and more precise: −1.9, −2.4, −2.1, −1.7, −1.25, −0.9, −0.7, −0.5.

So the effect peaks at 2 hours and then decays by roughly 25% per hour.

**What I will do next.** Test whether the effect is linear in dose. Use the one-hour finger outcome with rate dummies inside each band, controlling flexibly for the reading. Within a band, rate variation is exogenous noise. Then:
- compare the coefficient per extra unit/h at low and high rates;
- check whether the effect depends on the glucose level (by band).

**What would count as evidence.**
- Equal per-unit effects across rates and bands mean a linear, homogeneous effect, so ×0, ×0.5, ×1.5, ×2 and ×3 all scale with (multiplier − 1).
- Shrinking per-unit effects at high rates would mean saturation.

```python
d = xs[(xs.age_glucose_finger==0) & xs.gf_next.notna() & (xs.ins_rate>0)].copy()
d = d[~(d.age_glucose_lab.fillna(99) < d.age_glucose_finger)]
gfv = d.glucose_finger.values
ctrl = [gfv] + [np.maximum(gfv-k,0) for k in [165,180,200,220,240,260]] + [(d.band.values==k).astype(float) for k in [2,3,4]] + [d.I_l1.fillna(0).values, d.I_l2.fillna(0).values, d.lg_l1.fillna(d.glucose_finger).values]
for bnd in [1,2,3]:
    dd = d[d.band==bnd]
    gfv2 = dd.glucose_finger.values
    rates = sorted([r for r,c in dd.ins_rate.value_counts().items() if c>800])
    ref = {1:1.0,2:2.0,3:3.0}[bnd]
    dums = [(dd.ins_rate.values==r).astype(float) for r in rates if r!=ref]
    names = [f'r{r}' for r in rates if r!=ref]
    c2 = [gfv2, np.maximum(gfv2-(150+180)/2 if bnd==1 else gfv2-200,0), dd.I_l1.fillna(0).values, dd.I_l2.fillna(0).values, dd.lg_l1.fillna(dd.glucose_finger).values]
    res = ols(dd.gf_next.values, dums + c2, names+['g','gsp','il1','il2','lgl1'])
    out = res.loc[names].copy()
    out['per_unit'] = out.b / (np.array([float(n[1:]) for n in names]) - ref)
    print("band", bnd, "n", len(dd)); print(out.round(2).T.to_string())
```