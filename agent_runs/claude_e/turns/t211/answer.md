The insulin response grew a little more than linearly. Insulin ×2 went from −1.85 to −13.01 and insulin ×½ to +5.41. The NE readouts stayed at the original (−0.51 and −0.62), as they should without the NE part.

The stay set is not quite the one used in fwd_10, though: 60 insulin stays and 554 stay-hours here, against 55 and 543 there. So fwd_10 probably used `sel10.json`. I need my truth estimate on exactly these rows before judging the gain.

I'll compute the truth for each edit on the fwd_12 reached rows, along with its profile by hours since the first reached hour. I'll also check how `sel10` differs from 2020000000–99.

```python
de12 = pd.read_csv('/work/fwd_12/dose_edit_edited.csv'); do12 = pd.read_csv('/work/fwd_12/dose_edit_original.csv')
s10 = json.load(open('/work/sel10.json'))
print('sel10 n', len(s10), 'overlap with f3', len(set(map(int,s10)) & set(map(int,f3_ids))))
TRU12 = {}
for ed, fn, m in [('insulin_x2', truth_ins, 2.0), ('insulin_half', truth_ins, 0.5), ('ne_x2', truth_ne, 2.0), ('ne_half', truth_ne, 0.5)]:
    sub = de12[(de12['edit']==ed) & de12['reached']].copy()
    so = do12[(do12['edit']==ed) & do12['reached']]
    sub['truth'] = [fn(int(s), int(h), m) for s, h in zip(sub.stay_id, sub.hour)]
    sub['orig'] = so['shift'].values
    sub['k'] = sub.hour - sub.groupby('stay_id').hour.transform('min')
    TRU12[ed] = sub
    print(ed, 'n', len(sub), 'truth %.2f | edited %.2f | orig %.2f | ratio (ed-orig)/(truth-orig) %.3f' % (sub.truth.mean(), sub['shift'].mean(), sub.orig.mean(),
          (sub['shift'].mean()-sub.orig.mean())/(sub.truth.mean()-sub.orig.mean())))
g_ = TRU12['insulin_x2'].groupby('k')[['truth','shift']].mean().round(1).T
print(g_.to_string())
```