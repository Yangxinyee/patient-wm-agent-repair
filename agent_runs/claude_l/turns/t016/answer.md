Norepinephrine looks similar to insulin. The NE policy explains 93% of the rate variance (residual SD 0.09 overall, 0.14 while on NE), with steady titration downward as MAP rises. It passes the negative control: the residual does not predict same-hour cuff MAP given arterial MAP (0.09 ± 0.50). The residual raises the next hour's arterial MAP by 3.30 mmHg per mcg/kg/min (SE 0.09). So both drugs have real one-hour effects in the expected directions, and both have exogenous dosing noise to identify them.

What the evaluation needs, though, is the open-loop effect over hours 13–24, which depends on how persistent glucose and MAP are and whether the drug effect builds up or decays. Before building a dynamic model I'll look at the natural course of each variable:
- mean glucose and MAP by hour, split by treated vs untreated stays, to see trends;
- the lag-k autocorrelation of readings in untreated stays.

A near-random-walk latent process would mean effects accumulate hour after hour. Strong mean reversion would keep them near the one-hour size.

```python
dd = d.copy()
dd['ins_stay'] = st.has_ins.reindex(dd.stay_id).values
dd['ne_stay'] = st.has_ne.reindex(dd.stay_id).values
print(dd.groupby(['ins_stay','ih']).glucose_finger.mean().unstack(0).round(1).T)
print(dd.groupby(['ne_stay','ih']).map_arterial.mean().unstack(0).round(1).T)
# autocorr in untreated stays (no insulin) for finger glucose
w = P['glucose_finger'].unstack()  # stays x hours
noins = st.index[~st.has_ins]
wn = w.loc[noins]
for k in [1,2,3,4,6,8,12]:
    a = wn.iloc[:, :-k].values.ravel(); b_ = wn.iloc[:, k:].values.ravel(); ok = ~np.isnan(a)&~np.isnan(b_)
    print(k, ok.sum(), np.corrcoef(a[ok], b_[ok])[0,1].round(4), "mean diff sq/2:", (np.mean((a[ok]-b_[ok])**2)/2).round(1))
```