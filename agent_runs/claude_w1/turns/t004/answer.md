All stays last exactly 25.17 hours. Readings fall at whole hours, and each infusion record starts 5 minutes after the hour.
- Insulin rates come in 0.5 u/h steps.
- Norepinephrine rates come in 0.01 steps up to 1.0, which looks like a cap.
- About 61% of stays get insulin and 44% get norepinephrine.
- Arterial MAP appears in 48% of stays, and cuff MAP in nearly all.

Next I'll look at hour-by-hour coverage: which hours have each reading, and whether infusions run every hour once started. I'll also print a norepinephrine stay to see how MAP and doses interleave. This tells me how to set up the effect regressions.

```python
ev = tr.copy()
ev['h'] = np.floor(ev.hour).astype(int)
cov = ev.groupby(['event','h']).stay_id.nunique().unstack(fill_value=0)
print(cov.loc[:, -6:25].to_string())
# example norepi stay
ne_stays = has.index[(has.norepinephrine>0)&(has.map_arterial>0)]
sid = ne_stays[0]
print(tr[tr.stay_id==sid].drop(columns=['rate_unit','dose_unit']).to_string())
```