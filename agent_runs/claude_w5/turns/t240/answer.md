I can compare local and tool predictions in two ways. One is to reuse the train-stay head machinery (`fixed`, `TRr`, `probs_from_v0`) with a factual forward pass. The other is to check cheaply whether the tool's hour-13 offset, edited minus original factual prediction, tracks the recorded 12:05 dose and 12:00 reading on the tuning stays.

I'll start with the cheap check from the tuning CSV:
- If the offset at hour 13 rises with insulin rate, or is large even with no drug at 12:05, the injection or gate behaves differently from my design.
- If it sits near zero for drug-free stays and scales with the residual, the design works and the problem lies in the rollout dynamics.

```python
tun = pd.read_csv('/data/tuning.csv')
print(tun.columns.tolist()); print(tun.head(3).to_string())
selA = json.load(open('/work/sel_tuning_A.json')) if 'json' in globals() else None
import json; selA = json.load(open('/work/sel_tuning_A.json'))
print(len(selA), selA[:3])
fac = m.groupby(['stay_id','hour','edit']).first().reset_index()
g13 = de_o[(de_o.hour==13)&(de_o.edit=='insulin_x2')][['stay_id','factual']].merge(
      de_e[(de_e.hour==13)&(de_e.edit=='insulin_x2')][['stay_id','factual']], on='stay_id', suffixes=('_o','_e'))
g13['d'] = g13.factual_e - g13.factual_o
print(g13.d.describe())
```