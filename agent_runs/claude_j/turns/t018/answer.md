The titration rule has three zones on whichever MAP is read, map_arterial or map_cuff alone:

| MAP | rate change per hour (log ratio) |
|---|---|
| below 65 | rises, +0.03 to +0.18 (less at low MAP, likely because of the 1.0 cap) |
| 65 to 75 | flat, about -0.02 |
| above 75 | falls, about -0.2 to -0.3 |

Around these steps there is large multiplicative noise, SD about 0.29-0.35. The 65 and 75 thresholds and the noise are what I can exploit.

But I must find out whether the noise is a transient error around an intended rate or a persistent random-walk innovation, so that I can deconvolve the impulse response correctly. I also need to know which reading the clinician uses when both MAPs exist. Next I will:
1. look at the start probability near 65 at fine resolution for patients not yet on norepinephrine;
2. compare map_arterial and map_cuff as drivers when both are present;
3. compute the autocorrelation of the log-rate changes.

A lag-1 autocorrelation near -0.5 would mean transient noise. Near 0 would mean a random walk.

```python
z = P[(P.h>=0)].copy(); z['prevOn']=(z.N0_prev>0).astype(int)
zs = z[(z.prevOn==0)&z.MA.notna()]
f = zs.groupby(zs.MA.round()).anyN.agg(['size','mean'])
print(f.loc[58:72].T.round(3).to_string())
# both present: which drives start?
zb = z[(z.prevOn==0)&z.MA.notna()&z.MC.notna()]
for lab,cond in [('MA<65,MC>=66',(zb.MA<65)&(zb.MC>=66)),('MA>=66,MC<65',(zb.MA>=66)&(zb.MC<65)),('both<65',(zb.MA<65)&(zb.MC<65)),('both>=66',(zb.MA>=66)&(zb.MC>=66))]:
    print(lab, cond.sum(), zb[cond].anyN.mean().round(3))
# autocorrelation of log-rate changes
z = P[(P.h>=1)].copy()
z['lr'] = np.where((P.N0_prev>0)&P.N.notna(), np.log(P.N/P.N_prev), np.nan)
z['lr_prev'] = z.groupby('stay_id').lr.shift(1)
zz = z[['lr','lr_prev']].dropna()
print("autocorr lr:", zz.corr().iloc[0,1], len(zz))
# by MAP band 65-75 only (no systematic titration)
zm = z[(z.MA>65)&(z.MA<75)]
zz2 = zm[['lr','lr_prev']].dropna(); print("autocorr lr in mid band:", zz2.corr().iloc[0,1], len(zz2))
```