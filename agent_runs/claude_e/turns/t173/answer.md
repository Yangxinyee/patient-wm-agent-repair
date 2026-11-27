The mismatch is just z-draw randomness. A different seed changes predicted means by about 0.7 on average and leaves the mean NLL almost unchanged. So offline costs must compare edited and original heads with the same seed (paired), and must be cross-fitted so readouts are scored on stays they weren't fit on.

Before building that, I want the NE rollout profile from fwd_10, like the one I made for insulin. The mean shift by hours since the first NE dose from hour 12 shows how strongly the model carries a MAP shift forward. Comparing it with a truth profile from my estimated NE response curve shows whether a level correction (acting every hour) or a change correction (acting once) fits the truth's time course.

If the model's carry-forward is weak, as for glucose (about 0.35 per hour), a level-type correction applied every hour is needed to match the truth's build-up.

```python
IR_ne = np.array([3.36, 2.55, 1.84, 0.94, 0.50, 0.20])
neGt = Gtu['ne']
def truth_ne(s, h, mult):
    row = neGt.loc[s]; tot = 0.0
    for j in range(1, 7):
        t = h - j
        if t >= 12 and t in row.index and pd.notna(row[t]): tot += IR_ne[j-1]*(mult-1)*row[t]
    return tot
for edit, mult in [('ne_x2', 2.0), ('ne_half', 0.5)]:
    for nm, df in [('orig', d10o), ('edit', d10e)]:
        r = df[(df.edit==edit) & df.reached].copy()
        r['fd'] = r.stay_id.map({s: first_dose_hour(s, neGt) for s in r.stay_id.unique()})
        r['k'] = r.hour - r.fd
        r['tru'] = [truth_ne(s, h, mult) for s, h in zip(r.stay_id, r.hour)]
        g = r.groupby(r.k.clip(upper=8))[['shift','tru']].mean()
        print(edit, nm, 'model k=1..8+', g['shift'].round(2).tolist(), '| truth', g['tru'].round(2).tolist(), '| means %.2f %.2f' % (r['shift'].mean(), r['tru'].mean()))
r = d10e[(d10e.edit=='ne_x2') & d10e.reached]
print('mean recorded NE rate at h-1 on reached rows: %.3f' % np.mean([neGt.at[s, h-1] if pd.notna(neGt.at[s,h-1]) else 0 for s,h in zip(r.stay_id, r.hour)]))
```