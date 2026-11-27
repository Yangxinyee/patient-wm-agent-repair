The error is the same name clash as before: `observed` is also a groupby attribute. I will index that column by its name and rerun the same split by glucose level.

```python
g['prev']=g.groupby('stay_id')['observed'].shift(1)
g['pb']=pd.cut(g.prev,[0,110,130,150,180,220,400])
print(g.groupby(['grp','pb'],observed=True).agg(n=('dp','size'),dp=('dp','mean'),dnll=('d','mean')).round(3))
```