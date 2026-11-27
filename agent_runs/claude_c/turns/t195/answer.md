Head-0 ×4's cost is spread across codes. It includes systematic mean shifts at readings with no drug involved (lab glucose +0.80, arterial MAP +0.29). That points to a constant offset the amplified head adds at every bundle, not only to the amplified drug effect. Next I will split the finger-glucose changes by whether insulin ran in the hour before the reading, to see how much of the change is selective.

```python
ins=tu[tu.event=='insulin'][['stay_id','hour','rate']].copy(); ins['h']=np.floor(ins.hour).astype(int)+1
g=m[m.event=='glucose_finger'].copy(); g['h']=g.hour.astype(int)
g=g.merge(ins[['stay_id','h','rate']],on=['stay_id','h'],how='left'); g['rate']=g.rate.fillna(0)
g['grp']=np.where(g.rate>0,'after_ins','no_ins')
print(g.groupby('grp').agg(n=('d','size'),dnll=('d','mean'),dp=('dp','mean'),adp=('dp',lambda x: x.abs().mean())).round(4))
print(np.polyfit(g.rate[g.rate>0], g.dp[g.rate>0],1).round(3))
```