Head-0 ×4 is partly selective. After insulin, finger-glucose NLL improves slightly (−0.0005) and predictions drop in proportion to the rate (about −0.69 per unit rate). Without insulin, predictions rise by about 1.8 and NLL worsens by 0.018. Next I will check whether that rise depends on the glucose level (head 0 also reading the glucose token) or is a constant offset.

```python
g=g.sort_values(['stay_id','h']); g['prev']=g.groupby('stay_id').observed.shift(1)
g['pb']=pd.cut(g.prev,[0,110,130,150,180,220,400])
print(g.groupby(['grp','pb'],observed=True).agg(n=('dp','size'),dp=('dp','mean'),dnll=('d','mean')).round(3))
```