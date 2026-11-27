Under the provisional insulin response, the true readouts would be about −14 to −15 mg/dL for insulin ×2 (pooled over stay-hours; −13 to −14 as a mean of stay means) and about +7 for ×½. Those halving numbers simply assume linearity, which I still need to check. Tuning has 230 stays with insulin from hour 12 on, 136 with norepinephrine, and 81 with both.

Before spending a forward pass, I need the actual stay IDs, and I want to start on norepinephrine. I will:
- print a set of 100 tuning stays: all 81 with both drugs, plus insulin-only and norepinephrine-only stays to make 100;
- map the norepinephrine policy the way I did for insulin: which MAP reading it uses (arterial or cuff), the on/off rule, how the rate depends on MAP, and whether the previous rate matters.

If the norepinephrine policy is also a memoryless function of a recorded reading plus random jitter, the same IV design will work. If it depends on the previous rate (titration), the instruments must account for that.

```python
rng = np.random.default_rng(0)
both_ids = ids_t[has_ins & has_ne]
ins_only = ids_t[has_ins & ~has_ne]; ne_only = ids_t[~has_ins & has_ne]
sel = list(both_ids) + list(rng.choice(ins_only, 10, replace=False)) + list(rng.choice(ne_only, 9, replace=False))
sel = [int(x) for x in sel]
json.dump(sel, open('/work/sel100.json','w'))
print(len(sel)); print(json.dumps(sel))
# Norepi policy
pn0 = pn0.sort_values(['stay_id','h'])
g0 = pn0.groupby('stay_id')
pn0['ne_prev'] = g0.norepinephrine.shift(1).fillna(0)
fa = pn0.map_arterial.notna(); fc = pn0.map_cuff.notna()
for nm, mask, val in [('art only', fa&~fc, pn0.map_arterial), ('cuff only', fc&~fa, pn0.map_cuff), ('both-art', fa&fc, pn0.map_arterial), ('both-cuff', fa&fc, pn0.map_cuff), ('none', ~fa&~fc, None)]:
    sub = pn0[mask]
    if val is None:
        print(nm, len(sub), " p_on:", (sub.norepinephrine>0).mean().round(3)); continue
    v = val[mask]
    tab = sub.assign(v=v).groupby(pd.cut(v,[0,50,55,58,59,60,61,62,65,70,75,80,85,90,200]), observed=True).agg(n=('norepinephrine','size'),
            p_on=('norepinephrine', lambda x:(x>0).mean()), rate=('norepinephrine', lambda x: x[x>0].mean()),
            p_on_prev=('ne_prev', lambda x:(x>0).mean()))
    print(nm); print(tab.round(3).T.to_string())
```