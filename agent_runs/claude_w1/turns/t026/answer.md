In the tuning set, 215 of 400 stays receive insulin from hour 12 on and 141 receive norepinephrine (79 of them with arterial MAP). The typical rate after hour 12 is about 1.56 u/h insulin and 0.46 mcg/kg/min norepinephrine. Under my estimates, doubling insulin should therefore lower glucose by roughly 13–20 mg/dL. Doubling norepinephrine should raise MAP by about 3–4 mmHg.

For the first forward pass I'll choose 100 tuning stays with drug exposure after hour 12, putting stays with both drugs first, since each dose_edit request covers all four edits. I'll save the list so later passes can reuse the same stays.

```python
info['both'] = (info.ins12>0)&(info.ne12>0)
info['any'] = (info.ins12>0)|(info.ne12>0)
print(pd.crosstab(info.ins12>0, info.ne12>0))
print(pd.crosstab(info.ne12>0, info.art>0))
setA = list(info[info.both].index) 
rest_ins = list(info[(info.ins12>0)&~info.both].index)
rest_ne = list(info[(info.ne12>0)&~info.both].index)
rng = np.random.default_rng(0)
need = 100-len(setA)
add = list(rng.choice(rest_ne, size=min(len(rest_ne), need//2), replace=False))
add += list(rng.choice(rest_ins, size=100-len(setA)-len(add), replace=False))
setA = [int(x) for x in setA+add]
print(len(setA), len(set(setA)))
json.dump(setA, open('/work/setA.json','w'))
print(json.dumps({"type":"dose_edit","model":"original","split":"tuning","stays":setA}))
```