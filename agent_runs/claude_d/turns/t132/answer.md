The error has structure: glucose readings whose previous bundle was not an insulin bundle are biased by +4.7. Overall RMS is about 8 for glucose_finger in every subgroup, against 1.5–2.7 for MAP. Two possible causes are left. First, some readings may be missing from the factual CSV, which would make my register wrong. Second, a register feature may be mapped to the wrong column. I'll check whether the CSV contains every measurement atom. I'll also regress the error on the register features, the gap, and the previous-bundle type: a feature with a strong coefficient is likely mis-specified.

```python
meas = ATS[(ATS.akind==0)&ATS.event.isin(list(NUMV.keys()))]
print('meas atoms', len(meas), 'csv', len(FR))
g_ = FR[FR.event=='glucose_finger'].copy()
g_['zl'] = zsc_('glucose_finger', g_['last_val'].fillna(164)); g_['la']=np.log1p(((g_['minute']-g_['last_min']).fillna(0))*60)/10
g_['zo'] = zsc_('glucose_finger', g_['observed'])
Xf = np.column_stack([np.ones(len(g_)), g_['zl'], g_['la'], g_['has'].astype(float), g_['prev_ins'], g_['dtm']/60, g_['zo'], g_['predicted_mean']/52])
cf_,*_ = np.linalg.lstsq(Xf, g_['e'].values, rcond=None)
r2_ = 1-((g_['e']-Xf@cf_)**2).sum()/((g_['e']-g_['e'].mean())**2).sum()
print(np.round(cf_,2), 'R2', round(r2_,3))
print('corr e vs csv mean', np.corrcoef(g_['e'], g_['predicted_mean'])[0,1].round(3), 'sd csv mean', g_['predicted_mean'].std().round(1))
```