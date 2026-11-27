The bundle structure is clear:
- One bundle per distinct time.
- Infusion records produce an END atom one hour after they start, so the bundle at s + 1.083 holds the previous record's END and the next record's start.
- The admission records end in a final bundle at 25.1667.
- `has_insulin` marks a new insulin record (a WORLD atom), not an END.

Rebuilding the encoders exactly would mean guessing several categorical indices, so I'll first get more out of the data I already have. I'll compare the model's insulin x2 shift with my estimated true shift, row by row and by hours since the first dose. This tells me whether the model has:
- a correct short-lag effect but no persistence,
- a uniformly weak effect,
- or an effect unrelated to dose size.

That determines what kind of edit is needed.

```python
t2 = true_shift(D1, Gtu, 'insulin_x2', d_f)
# hours since first insulin >= 12 for each row
first = {sid: min([h for h in Gtu['ins'].loc[sid].dropna().index if h>=12], default=99) for sid in t2.stay_id.unique()}
t2['since'] = t2.hour - t2.stay_id.map(first)
print(t2.groupby('since').agg(n=('shift','size'), model=('shift','mean'), truth=('true_est','mean')).round(2).T.to_string())
print('slope model~truth:', np.polyfit(t2.true_est, t2['shift'], 1).round(3), ' corr', np.corrcoef(t2.true_est, t2['shift'])[0,1].round(3))
# dependence on current dose (dose at hour-1) for model shift
t2['dose_prev'] = [Gtu['ins'].loc[s][h-1] for s,h in zip(t2.stay_id,t2.hour)]
print(t2.groupby(t2.dose_prev.fillna(0)).agg(n=('shift','size'),model=('shift','mean'),truth=('true_est','mean')).round(2).to_string())
```