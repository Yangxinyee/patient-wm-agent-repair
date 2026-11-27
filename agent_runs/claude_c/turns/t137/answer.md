The dose-edit CSV holds per-stay-hour shifts with a 'reached' flag. I'll compute the mean shift by lag since the first reached hour, comparing ×3 against the original for each edit, to see whether the direct lag-1 response scaled.

```python
def bylag(df):
    r=df[df.reached].copy()
    r['lag']=r.hour-r.groupby(['stay_id','edit']).hour.transform('min')+1
    return r.groupby(['edit',np.minimum(r.lag,6)])['shift'].mean().unstack(0).round(2)
a=bylag(do); b=bylag(de)
print(pd.concat({'orig':a,'x3':b},axis=1).to_string())
```