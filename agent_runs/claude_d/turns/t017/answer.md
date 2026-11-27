The norepinephrine effect estimated from the titration noise is positive and builds up. Given current MAP and the previous rate, each +1 mcg/kg/min at hour t raises arterial MAP by about 2.9 mmHg at t+1, 4.7 at t+2 and 5.4 at t+3. Part of that build-up is the rate itself persisting. Within cells of MAP and previous rate the one-hour coefficient is fairly stable at 3–4.

The negative coefficient on the previous rate (−3.6) means current MAP already holds earlier norepinephrine's effect. This points to a lagged or accumulating response, so I will need a structural dynamic model, for example g-estimation with the policy residuals as instruments, to get effects with later doses held fixed.

Before refining, I want to see what the original model currently predicts and the exact output format. Next I will pick 100 tuning stays where insulin and/or norepinephrine run after hour 12 and save the list. The next reply will spend one forward pass on dose_edit for the original model.

```python
U = PUu.copy()
hidx = U.index.get_level_values('h')
ins12 = U[(hidx>=12)].groupby(level='stay_id').ins.apply(lambda x: x.notna().sum())
nor12 = U[(hidx>=12)].groupby(level='stay_id').nor.apply(lambda x: x.notna().sum())
s = pd.DataFrame({'ins12':ins12,'nor12':nor12})
print(len(s), (s.ins12>0).sum(), (s.nor12>0).sum(), ((s.ins12>0)&(s.nor12>0)).sum())
both_ = s[(s.ins12>0)&(s.nor12>0)].index.tolist()
ins_only = s[(s.ins12>0)&(s.nor12==0)].index.tolist()
nor_only = s[(s.ins12==0)&(s.nor12>0)].index.tolist()
rng = np.random.default_rng(0)
sel1 = sorted(rng.choice(both_, size=min(60,len(both_)), replace=False).tolist() + rng.choice(ins_only, 20, replace=False).tolist() + rng.choice(nor_only, 20, replace=False).tolist())
print(len(sel1), sel1[:5])
json.dump({'sel1':sel1}, open('/work/sel1.json','w'))
print(json.dumps(sel1))
```