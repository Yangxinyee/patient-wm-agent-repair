All 100 of those stays have insulin and norepinephrine in every hour, and both drugs arrive in the same hh:05 bundle. That sample can't separate the two drugs' representations, and it isn't typical. I'll instead draw a random sample of moderately drug-rich training stays, and check that it mixes insulin-only, norepinephrine-only and joint bundles.

```python
rng_=np.random.default_rng(7)
a_=df[df.event.isin(['insulin','norepinephrine'])].copy(); a_['hr']=a_.hour.round(3)
bt=a_.groupby(['stay_id','hr']).event.agg(lambda s: ''.join(sorted(set(x[0] for x in s))))
pool=cnt.index[(cnt.insulin+cnt.norepinephrine)>=10]
selS=list(rng_.choice(pool,100,replace=False))
mix=bt.loc[bt.index.get_level_values(0).isin(selS)].value_counts().to_dict(); print('pool',len(pool),'mix(i=ins,n=ne,in=both)',mix)
json.dump([int(x) for x in selS],open('/work/selS.json','w')); print(json.dumps([int(x) for x in selS]))
```