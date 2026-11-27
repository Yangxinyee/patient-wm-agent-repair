The titration table can't be read directly, because it doesn't hold the previous rate fixed. More informative are the placebo test and the flexible controls:

- **Placebo:** a shock does not "predict" MAP two hours before it (−0.03 ± 0.08), so the residual is not confounded by recent history.
- **Flexible controls:** controlling flexibly for the previous two MAP readings and the earlier rate leaves the negative association unchanged (k1 −0.32 ± 0.09).

I see one concrete artefact that would produce exactly this sign. My titration cells use 2-mmHg MAP bins starting at even numbers, so the bin [64, 66) may straddle the "titrate up below ~65" threshold. Within that bin, readings just under the threshold get positive residuals and readings just over it get negative ones. Because MAP persists, those positive residuals then sit on slightly lower later MAP, which is a spurious negative effect.

Next I'll check the resolution of the MAP readings and where the 65 and 75 thresholds fall, using exact values and avoiding the cap. Then I'll rebuild the titration residual with cells aligned to the thresholds plus fine MAP bins.

What would settle it:
- If the −0.3 effect disappears, it was this artefact, and zero is right for norepinephrine.
- If it survives, the small negative effect is more likely real.

```python
v=tr.loc[tr.event.isin(['map_arterial','map_cuff']),'value'].values
print('MAP frac parts (first 10 unique):', np.unique(np.round(v%1,3))[:10], 'n unique frac', len(np.unique(np.round(v%1,3))))
gv=tr.loc[tr.event.isin(['glucose_finger']),'value'].values
print('finger frac parts:', np.unique(np.round(gv%1,3))[:10])
mid=(prevN>0.1)&(prevN<0.7)&on
dd=pd.DataFrame({'lr':np.log(np.maximum(curN[mid],1e-9)/prevN[mid]),'m':M0[mid],'art':~np.isnan(Ar[:,1:][mid])}).dropna()
for lo_,hi_ in [(62,68),(72,78)]:
    sub=dd[(dd.m>=lo_)&(dd.m<hi_)]
    print(sub.groupby(np.floor(sub.m*2)/2)['lr'].agg(['mean','count']).round(3).T.to_string())
```