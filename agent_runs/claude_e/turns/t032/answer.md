The titration is now clear. When an arterial MAP is taken that hour, its band alone sets the step:

| arterial MAP band | mean log rate change |
|---|---|
| below 65 | +0.11 to +0.16 |
| 65–75 | about 0 |
| above 75 | about −0.2 |

The cuff reading makes no difference in those hours. When only a cuff reading exists, its band drives the step in the same way. In hours with no MAP reading at all the mean change is small (+0.06), and is probably based on the last reading. On top of the step there is large noise, with a log-ratio SD of about 0.3.

So, given the previous rate and the MAP reading the clinician acted on, the rate change is essentially random. That supports the same local-projection design as for insulin. I'll regress later arterial MAP on the current rate, with:
- fixed effects for the exact arterial reading at t,
- flexible controls for the previous rate,
- the sample restricted to continuing infusions,
- the previous-rate effect absorbed.

Before that, I'll check how map_arterial readings are scheduled (per-stay arterial lines or random hours) to rule out selection problems.

```python
ma_pres = Gtr['ma'].loc[:, -1:24].notna()
per_stay = ma_pres.mean(axis=1)
print('per-stay fraction of hours with arterial MAP:', per_stay.round(1).value_counts().sort_index().to_dict())
mc_pres = Gtr['mc'].loc[:, -1:24].notna(); print('cuff per stay:', mc_pres.mean(axis=1).round(1).value_counts().sort_index().to_dict())
print('corr(arterial frac, cuff frac) across stays:', np.corrcoef(per_stay, mc_pres.mean(axis=1))[0,1])
# within stays having arterial line: gaps
gaps=[]
for sid in per_stay.index[per_stay>0.5][:2000]:
    hh = np.where(ma_pres.loc[sid].values)[0]; gaps += list(np.diff(hh))
print('arterial gaps in high-art stays:', pd.Series(gaps).value_counts().head(6).to_dict())
# does presence depend on MAP level / NE?
L = Ltr; s = L[(L.h>=0)&(L.h<=23)]
print(s.groupby(pd.cut(s.ma_ff,[0,55,65,75,85,200]),observed=True).apply(lambda d: pd.Series({'n':len(d),'p_art_next':d.ma_n1.notna().mean(),'p_cuff_next':d.mc_n1.notna().mean()})).round(3).to_string())
```