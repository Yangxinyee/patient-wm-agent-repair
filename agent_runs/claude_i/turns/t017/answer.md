Two things are now clear about the readout:
- Rows exist only at hours with a recorded finger-glucose reading (for insulin) or arterial MAP reading (for norepinephrine).
- A row is "reached" once a dose has run since hour 12.

For stay 2020000000 (rates 0.5–1.5 from hour 15), the model's ×2 shift is about −1.5 at the first reachable hour and stays near −2 to −3 later. Its ×½ shift is almost zero there.

To judge whether −2.0 is right, I need the true time course of insulin's effect: the kernel K_j, the effect of one dose j hours later with later doses held fixed. That calls for a state-space model of latent glucose. The policy and the measurement schedule depend only on recorded data, so likelihood-based estimation is valid despite feedback and selection.

Before building it, I'll look at the baseline glucose dynamics:
- mean glucose by hour, to check for a trend;
- persistence in stays that never get insulin;
- whether readings are scheduled periodically or at random when off insulin.

```python
# baseline dynamics
gfm = P.groupby(level='h')[['gf','gl']].mean()
print("mean gf/gl by hour:", gfm.round(1).T.to_string())
noins = ins_stays[ins_stays==0].index
Pn = P.loc[noins]
print("no-insulin stays:", len(noins), " mean gf", Pn.gf.mean().round(1), " sd", Pn.gf.std().round(1))
# autocorrelation of g_any at lags in no-insulin stays (pairs where both present)
for lag in [1,2,3,4,6,8,12]:
    a = Pn.g_any; b = Pn.groupby(level='stay_id').g_any.shift(-lag)
    m = a.notna()&b.notna()
    print(lag, "corr %.3f  mean diff %.2f  sd diff %.2f  n %d" % (np.corrcoef(a[m],b[m])[0,1], (b[m]-a[m]).mean(), (b[m]-a[m]).std(), m.sum()))
# stay-level mean variance
sm_ = Pn.groupby(level='stay_id').g_any.mean()
print("sd of stay means (no-insulin):", sm_.std().round(1))
# measurement schedule when not on insulin: is gf presence random? autocorr of presence in no-ins stays
pres = Pn.gf.notna().astype(float)
pres_prev = pres.groupby(level='stay_id').shift(1)
mm = pres_prev.notna()
print("P(gf_t | gf_{t-1}) =", pres[mm & (pres_prev==1)].mean().round(3), " P(gf_t | no gf_{t-1}) =", pres[mm & (pres_prev==0)].mean().round(3))
print("gl presence prob by hour (all):", P.gl.notna().groupby(level='h').mean().round(2).values)
```