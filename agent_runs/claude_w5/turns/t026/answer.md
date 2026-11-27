My last reply contained two code blocks, so nothing ran. The plan is unchanged.

First I'll select an informative set of 100 tuning stays in Python:
- stays with insulin from hour 12 and norepinephrine with arterial readings;
- then norepinephrine-only stays with arterial readings;
- then insulin-only stays.

In the next reply I'll send the dose_edit forward request for that list.

```python
both_ids = sorted(set(tu_ins12.index) & set(tu_ne12.index) & set(tu_art.index))
ins_only = sorted(set(tu_ins12.index) - set(both_ids))
ne_only = sorted((set(tu_ne12.index) & set(tu_art.index)) - set(both_ids))
sel = both_ids[:]
k1 = (100 - len(sel))//2
sel += ne_only[:k1]
sel += ins_only[:100-len(sel)]
sel = [int(x) for x in sel]
print(len(both_ids), len(ne_only), len(ins_only), len(sel))
json.dump(sel, open('/work/sel_tuning_A.json','w'))
print(json.dumps(sel))
```