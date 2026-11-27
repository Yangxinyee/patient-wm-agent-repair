The norepinephrine policy has a clear outline:
- It starts only when the drug was off and a fresh MAP is at or below about 64, always at about 0.05 (sd 0.015). It never starts without a fresh MAP or above 65.
- While on, the rate is multiplied each hour by a noisy factor: about 1.1–1.2 when MAP is below 65, about 1.0 at 65–75, and about 0.83 above 75, with sd 0.3–0.45. Above 75 there is a 15–20% chance of stopping.

The example stay worries me, though. The rate fell steadily from 1.0 to 0.43 over six hours while the arterial MAP was 48–58, then rose. The hourly changes look autocorrelated, with runs of decreases and increases, rather than independent noise around a function of the current reading. That would happen if the titration follows something smooth that is not in the current reading: an autocorrelated clinician noise (harmless), or a hidden patient state (a confounder).

Next I will test this directly:
- the lag-1 autocorrelation of the log rate change;
- a regression of the log change on the current and previous MAP, the previous rate, the hour and the previous log change;
- how arterial and cuff MAP compare when taken together.

Strongly autocorrelated changes not explained by recorded MAP history would mean the instrument logic needs care.

```python
x = x.sort_values(['stay_id','h'])
gx = x.groupby('stay_id')
x['lr'] = np.where((x.nr>0)&(x.ne_prev>0), np.log(x.nr.clip(lower=1e-3)/x.ne_prev.clip(lower=1e-3)), np.nan)
x['lr_prev'] = gx.lr.shift(1)
x['map_prev'] = gx.map_pol.shift(1)
x['map_art_last'] = gx.map_arterial.ffill(); x['map_pol_last'] = gx.map_pol.ffill()
ok = x.lr.notna() & x.lr_prev.notna()
# exclude clipped at 1.0 boundaries to avoid mechanical effects
okc = ok & (x.nr<1.0) & (x.ne_prev<1.0) & (x.nr>0.01)
print("lag1 autocorr of log rate change (all):", np.corrcoef(x.lr[ok], x.lr_prev[ok])[0,1].round(3), " (unclipped):", np.corrcoef(x.lr[okc], x.lr_prev[okc])[0,1].round(3))
# regression with fresh MAP
r = x[okc & x.map_pol.notna()].copy()
r['mapc'] = np.clip(r.map_pol, 40, 100)
Xc = np.column_stack([np.ones(len(r)), r.mapc, (r.mapc-65).clip(lower=0), (r.mapc-75).clip(lower=0), np.log(r.ne_prev), r.lr_prev, r.h])
b = np.linalg.lstsq(Xc, r.lr.values, rcond=None)[0]
res = r.lr.values - Xc@b
print("coef [1, map, (map-65)+, (map-75)+, log prev rate, lr_prev, h]:", b.round(4), " R2:", (1-res.var()/r.lr.var()).round(3))
# add previous MAP
r2 = r[r.map_prev.notna()].copy()
X2 = np.column_stack([np.ones(len(r2)), r2.mapc, (r2.mapc-65).clip(lower=0), (r2.mapc-75).clip(lower=0), np.log(r2.ne_prev), r2.lr_prev, np.clip(r2.map_prev,40,100)])
b2 = np.linalg.lstsq(X2, r2.lr.values, rcond=None)[0]; res2 = r2.lr.values - X2@b2
print("with map_prev: ", b2.round(4), " R2:", (1-res2.var()/r2.lr.var()).round(3))
# arterial vs cuff simultaneous
sim = pn0[pn0.map_arterial.notna() & pn0.map_cuff.notna()]
dd = sim.map_cuff - sim.map_arterial
print("cuff - art: mean", dd.mean().round(2), " sd", dd.std().round(2), " n", len(sim))
print(sim.assign(dd=dd).groupby(pd.cut(sim.map_arterial,[0,50,60,70,80,90,200]), observed=True).dd.agg(['mean','std','size']).round(2).to_string())
```