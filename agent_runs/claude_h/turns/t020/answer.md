The on/off discontinuity at 150 shows no glucose jump: −0.21 ± 0.31 at 1 hour, despite a dose jump of 0.83 u/h. The pooled RD over the 180–300 thresholds gives per-unit effects of −0.07 ± 0.13, +0.03 ± 0.14, −0.14 ± 0.16, +0.10 ± 0.17 and −0.06 ± 0.20 at horizons 1, 2, 3, 4 and 6. (The error was only a missing 8-hour column.) Every design agrees: insulin does not change finger glucose here, so the true x2 and x1/2 insulin readouts should be about 0.

Next I need to see what the original model currently predicts. I will choose tuning stays with drug exposure after hour 12 for the dose_edit request: preferably stays with both insulin and norepinephrine from hour 12 on, topped up with stays having either. I will then spend one forward pass on dose_edit with the original model, which gives the baseline for all four readouts. I expect the model to show confounded effects; for insulin x2 probably a glucose rise, since higher insulin co-occurs with higher glucose.

```python
PT = PT.copy()
PT['ins']=PT.insulin.fillna(0); PT['nor']=PT.norepinephrine.fillna(0)
late = PT.reset_index()
late = late[late.ih>=12]
s_ins = late.groupby('stay_id').ins.apply(lambda x:(x>0).sum())
s_nor = late.groupby('stay_id').nor.apply(lambda x:(x>0).sum())
both = s_ins.index[(s_ins>0)&(s_nor>0)]
only_i = s_ins.index[(s_ins>0)&(s_nor==0)]
only_n = s_ins.index[(s_ins==0)&(s_nor>0)]
print(len(both), len(only_i), len(only_n), "none", ((s_ins==0)&(s_nor==0)).sum())
rng = np.random.default_rng(0)
sel = list(both[:60]) + list(only_i[:20]) + list(only_n[:20])
print(len(sel))
import json as _j
print(_j.dumps({"type":"dose_edit","model":"original","split":"tuning","stays":[int(x) for x in sel]}))
with open('/work/sel1.json','w') as f: _j.dump([int(x) for x in sel], f)
```