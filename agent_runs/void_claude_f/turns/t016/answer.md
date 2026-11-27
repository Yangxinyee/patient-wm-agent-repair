Every check comes back null. Within each sliding-scale step, the insulin rate coefficient on glucose two hours later is between −0.11 and +0.17 mg/dL per U/h, all within about 1.2 SE of zero. On finger and lab glucose and at every lag up to 12 hours, the coefficients are a few hundredths to a tenth of a mg/dL per U/h, all within about 1.5 SE of zero.

The binned-residual means jump around without any trend. That pattern comes from step composition, because the discrete rates map to different residual values at each step.

If each unit of insulin moved glucose by even 1 mg/dL, I would see coefficients of about −1, many SE from zero. So my current belief is that the insulin rate has essentially no causal effect on glucose here, and the true x2 and x1/2 glucose effects are near 0 (within about ±2 mg/dL).

Before norepinephrine, I will record this in the notebook and run one more insulin test: compare glucose over the following hours for finger readings just below 150 (no insulin) vs just above (insulin starts). That tests whether insulin does anything at all. An on/off effect without a rate effect would suggest saturation; no effect at either margin would support "insulin is inert".

```python
with open('/work/notes.md','a') as f:
    f.write("""
## Insulin policy & effect (runs 6-15)
- Insulin given iff glucose reading at hour h (finger, else lab) >= 150 (unrounded). Rate = base(G) * exp(eps), rounded to 0.5,
  base: [150,180)->1, [180,220)->2, [220,260)->3, [260,300)->4, >=300->6. eps iid N(0,~0.33): lag-1 autocorr 0.005,
  no stay component, uncorrelated with same-hour lab glucose given finger => as-if randomized dose variation.
- Regress future finger/lab glucose (lags 1..12h) on insulin rate, controlling G_h + step dummies: coef ~0 (+-0.04..0.1 mg/dL per U/h).
  => INSULIN RATE HAS ~NO CAUSAL EFFECT ON GLUCOSE (tentative). True x2, x1/2 effects ~ 0.
""")
# Regression discontinuity at 150: finger G at h in [140,160], no insulin before h (to look at initiation)
xx['ins_prev_any'] = xx.groupby('stay_id').insulin.transform(lambda s: s.notna().cumsum().shift(1).fillna(0))>0
rd = xx[xx.glucose_finger.between(135,165) & (xx.h>=0) & (xx.h<=18)].copy()
rd['above'] = (rd.glucose_finger>=151).astype(float)
rd = rd[rd.glucose_finger!=150]
rd['gc'] = rd.glucose_finger - 150.5
for k in [1,2,3,4,6]:
    for nm in ['Gf','Gl']:
        d = rd[rd[f'{nm}_{k}'].notna()]
        X = np.column_stack([d.above, d.gc, d.gc*d.above])
        be, se = ols(d[f'{nm}_{k}'].values, X)
        print(f"RD {nm} k={k}: jump {be[1]:.2f} +- {se[1]:.2f}  (n={len(d)}) ins rate above mean {d[d.above==1].insulin.mean():.2f}")
```