The layout is clear. Readings fall on integer hours, and insulin records come at hh:05 with rate, dose and end_hour. I still need the norepinephrine timing and which stays carry many drug bundles. I'll check the norepinephrine timing, then pick 100 training stays rich in insulin and norepinephrine. The policy residuals (EPS) are already computed for training stays, so on those stays the states can be regressed on the dose surprise. The list is saved to /work for the forward request.

```python
ne_=df[df.event=='norepinephrine']; print('NE frac-hour', (ne_.hour%1).round(3).value_counts().head(4).to_dict(), 'n',len(ne_))
cnt=df[df.event.isin(['insulin','norepinephrine'])].groupby(['stay_id','event']).size().unstack(fill_value=0)
cnt['score']=cnt['insulin']+2*cnt['norepinephrine']
selS=cnt.sort_values('score',ascending=False).index[:100].tolist()
print(cnt.loc[selS,['insulin','norepinephrine']].describe().loc[['mean','min','max']].round(1).to_string())
json.dump([int(x) for x in selS],open('/work/selS.json','w'))
print(json.dumps([int(x) for x in selS]))
```