With column 0 set to the insulin flag, the glucose bias is about 0, but the median error stays at 3.6. Adding the insulin rate's z-score or bin does not help. The per-row optimum reached 0.54, so column 0 is continuous rather than a flag.

Within rows that had insulin in the last hour, I'll check how the best column-0 value relates to:
- the last insulin rate and its z-score;
- the age of the last insulin dose;
- the log-age transforms in seconds and minutes.

A near-perfect correlation with one of these identifies the feature.

```python
w_ = lia_ <= 1.0
cands = {'lir':lirG[w_], 'zr':zr_ins[w_], 'lia':liaG[w_], 'log1p_s/10':np.log1p(liaG[w_]*3600)/10, 'log1p_min/10':np.log1p(liaG[w_]*60)/10,
         'zl':cand['zl'][w_], 'age_h':cand['age_h'][w_]}
for k_,v_ in cands.items(): print(k_, 'corr', np.corrcoef(v_, bestv[w_])[0,1].round(3), 'mean', np.mean(v_).round(3))
import collections
print(collections.Counter(np.round(lirG[w_],1)).most_common(6))
for r__ in [0.5,1.0,1.5,2.0,3.0,4.0]:
    ww = w_ & (np.abs(np.nan_to_num(lirG)-r__)<0.01)
    if ww.sum()>5: print('rate', r__, 'n', ww.sum(), 'bestv med', np.median(bestv[ww]).round(2))
```