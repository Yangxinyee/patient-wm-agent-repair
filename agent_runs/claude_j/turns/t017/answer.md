Norepinephrine is a sticky titration.

- **Starting:** it starts when MAP falls below about 65 (88% at 60-65, about 100% below 60). It rarely starts above 70.
- **Continuing:** once on, it continues almost regardless of MAP: 98-99% up to 75, about 90% even above 80. Stops are rare.
- **Rate level:** the rate falls with rising MAP and climbs again at high MAP, presumably because the rate carries over from earlier hours.
- **Which MAP drives it:** map_cuff seems to stand in when map_arterial is missing, with the same pattern. With no MAP reading, the infusion simply continues.

Next I will model the titration step: the change, or log ratio, of the rate from the previous hour against the current MAP. I will find which reading is used when both exist, and the noise around the rule. A rule of the form rate_t = rate_{t-1} × step(MAP_t) × noise would allow a surprise instrument as for insulin, and possibly discontinuities at MAP thresholds.

```python
z = P[(P.h>=1)&(P.N0_prev>0)&P.N.notna()].copy()
z['lr'] = np.log(z.N/z.N_prev)
z['dN'] = z.N - z.N_prev
# use MA when available else MC
z['src'] = np.where(z.MA.notna(),'MA',np.where(z.MC.notna(),'MC','none'))
for src in ['MA','MC','none']:
    zz = z[z.src==src]
    if src=='none':
        print("none:", zz.lr.describe().round(3).to_dict()); continue
    col = src
    t = zz.groupby(pd.cut(zz[col],[0,45,50,55,60,62,64,65,66,68,70,72,74,75,76,78,80,85,90,200])).agg(n=('lr','size'), mean_lr=('lr','mean'), sd_lr=('lr','std'), mean_dN=('dN','mean'))
    print(src); print(t.round(3).to_string())
both = z[z.MA.notna()&z.MC.notna()]
print("both present n=",len(both))
```