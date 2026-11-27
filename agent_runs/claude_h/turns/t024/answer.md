Another name collision: `shift` is a pandas method, so `x.shift` returned the method. I will use bracket indexing and rerun the same summary of the original model's dose-edit output:
- mean shift with a stay-clustered SE;
- the hour profile;
- the spread across stays;
- the correlation between edits.

```python
r = de[de.reached]
for e in r.edit.unique():
    x = r[r.edit==e]
    sm_ = x.groupby('stay_id')['shift'].mean()
    print(f"{e}: mean over stay-hours {x['shift'].mean():+.3f}; mean of stay means {sm_.mean():+.3f} (se {sm_.std()/np.sqrt(len(sm_)):.3f}); n stays {len(sm_)}; sd stay means {sm_.std():.3f}")
    print("   by hour:", x.groupby('hour')['shift'].mean().round(2).to_dict())
    print("   median path se:", x.shift_se.median().round(3))
w = r.pivot_table(index=['stay_id','hour'], columns='edit', values='shift')
print(w.corr().round(3))
print(de.groupby('edit').reached.mean())
```