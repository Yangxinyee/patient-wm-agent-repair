The rollout fails inside `rollout_stay` itself, at its line 54. The only overwritten globals that any function reads are `n0` (used by `dnll`) and `R` (used by `head_probs`). `R` was never actually reassigned, because both rollout attempts failed. So the clobbering didn't cause this. More likely `rollout_stay` is an older version whose helper globals were later redefined; there is also a newer `rollout_gen(df, s, P, specs, ...)`. I'll list the globals each of the two rollout functions reads, with each one's type, to find the list being indexed by a string. I'll also note which constants `rollout_gen` expects for `specs`, so I can call the newer one if the old one is stale.

```python
for nm_ in ['rollout_stay','rollout_gen','prep_stay','true_shift']:
    co=globals()[nm_].__code__
    gn=sorted(x for x in gnames(co) if x in globals())
    print(nm_, 'vars:',co.co_varnames[:co.co_argcount], '| globals:', ', '.join('%s:%s'%(x,type(globals()[x]).__name__[:5]) for x in gn if not callable(globals()[x]) or x in ('torch','np'))[:600])
consts=[c for c in rollout_gen.__code__.co_consts if isinstance(c,(str,float,int)) and c is not None][:30]
print('rollout_gen consts', consts)
```