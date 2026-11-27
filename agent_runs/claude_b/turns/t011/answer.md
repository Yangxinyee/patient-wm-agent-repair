The norepinephrine policy is persistent titration. With an infusion running, the next rate rises 5–13% per hour (log scale) when MAP is below 64, stays about flat at 64–76, and drops about 25% when MAP is above 76. Hour-to-hour noise is about 0.3 on the log scale. Starting depends on MAP: about 55% start at MAP below 48 and almost none above 68. Infusions also continue in 26% of hours with no MAP recorded. So the norepinephrine dose depends on its own history and on MAP, with randomness I may be able to use.

Before estimating effects in depth, I want a baseline of the original model's dose answers. I will pick tuning stays that actually receive insulin or norepinephrine from hour 12 on, since only those enter the readout, and list them for a dose_edit request. I also set up a helper to build hourly panels for any split.

```python
Pt = panel(tu)
Pt = Pt.loc[(slice(None), slice(-1,24)), :].copy()
Pt['ins'] = Pt['insulin'].fillna(0.0); Pt['ne'] = Pt['norepinephrine'].fillna(0.0)
hrs = Pt.index.get_level_values(1)
late = Pt[hrs>=12]
ins_late = late.groupby(level=0)['ins'].apply(lambda x:(x>0).any())
ne_late = late.groupby(level=0)['ne'].apply(lambda x:(x>0).any())
print("tuning stays:", len(ins_late), " insulin>=12:", ins_late.sum(), " NE>=12:", ne_late.sum(), " both:", (ins_late&ne_late).sum())
both = ins_late[ins_late & ne_late].index.tolist()
ins_only = ins_late[ins_late & ~ne_late].index.tolist()
ne_only = ne_late[ne_late & ~ins_late].index.tolist()
sel1 = (both + ins_only[:25] + ne_only[:25])[:100]
print(len(both), len(ins_only), len(ne_only), len(sel1))
print(json.dumps([int(s) for s in sel1]))
```