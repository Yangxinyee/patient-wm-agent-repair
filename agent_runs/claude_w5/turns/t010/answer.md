The insulin policy is a sliding scale with random noise, which is good news for identification:
- glucose below 150 → no insulin (apart from 3–6% of readings at 140–150, to check);
- 150–180 → base rate about 1 u/h; 180–220 → about 2 u/h; 220–250 → about 3 u/h; above 250 → about 3.5–4;
- on top of the base rate, multiplicative log-normal noise with sd about 0.31–0.37, rounded to 0.5;
- once insulin has started it is always given above 150; before that, only 50–80% of eligible hours get a start.

Within a glucose band, then, the rate varies randomly. The band thresholds (180, 220, 250) also make the base rate jump, which gives a regression-discontinuity design. Both are sources of exogenous dose variation, unless the noise depends on hidden patient state; I'll test that by checking whether rate residuals predict anything about the past.

Next I'll look at the measurement side: finger vs lab glucose at the same hour (bias, noise), and finger and arterial autocorrelation. That tells me how to model the latent state and the dynamics I'll need for the multi-hour readout.

```python
both = P[P.glucose_finger.notna() & P.glucose_lab.notna()]
d = both.glucose_finger - both.glucose_lab
print("finger-lab paired n", len(both), "mean diff", d.mean(), "sd diff", d.std())
print(np.polyfit(both.glucose_lab, both.glucose_finger, 1))
bm = P[P.map_arterial.notna() & P.map_cuff.notna()]
dm = bm.map_cuff - bm.map_arterial
print("cuff-art paired n", len(bm), "mean diff", dm.mean(), "sd", dm.std(), np.polyfit(bm.map_arterial, bm.map_cuff,1))
# autocorrelation of hourly finger glucose differences
gf = P.pivot(index='stay_id', columns='h', values='glucose_finger')
dg = gf.diff(axis=1)
print("sd of hourly finger diff:", np.nanstd(dg.values))
x = dg.values[:, 1:-1].ravel(); y = dg.values[:, 2:].ravel(); m = ~np.isnan(x)&~np.isnan(y)
print("lag1 autocorr of differences:", np.corrcoef(x[m], y[m])[0,1])
```