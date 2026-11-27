The groupby failed because `shift` is a DataFrame method name, so attribute access returned the method. I will index the column by its name instead and rerun the same comparison.

```python
def prof(k,edit='insulin_x2'):
    out={}
    for f in sorted(glob.glob(f'/work/fwd_{k}/dose_edit_*.csv')):
        df=pd.read_csv(f); df=df[(df.edit==edit)&(df.reached.astype(bool))]
        out[f.split('_')[-1][:-4]]=df.groupby('hour')['shift'].mean().round(2).values
    return out
for k in [1,6,18]:
    print(k, prof(k))
```