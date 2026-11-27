`tensors.json` only gives shapes, so a full reimplementation of the core would leave many details to guess. Activations, atom construction and the time-mixing are all unspecified.

A more controllable plan is to edit only the value head, `attr_heads.value_bin`. Head weights don't feed back into the state in teacher forcing, so with the tool's `states` I could compute the edited model's factual NLL exactly offline. Only the rollout effect would need the tool.

Whether a head-only add-on can work depends on two questions:
1. Do the states encode the recent dose history linearly, so the head can read off the needed shift?
2. What one-step dose sensitivity has the model actually learned? Its 20% response at hour 13 suggests very little.

I'll use two forward passes on the same 100 training stays, chosen to have plenty of insulin and norepinephrine with arterial lines. This one is a factual run for per-reading predicted means and NLLs. Then I'll request states.

```python
rng = np.random.default_rng(0)
cnt_ins = P.groupby(level='stay_id')['ins'].count(); cnt_ne = P.groupby(level='stay_id')['ne'].count(); cnt_ma = P.groupby(level='stay_id').ma.count()
A = cnt_ins[(cnt_ins>=8)&(cnt_ne>=8)&(cnt_ma>=20)].index.values
B_ = cnt_ins[(cnt_ins>=8)&(cnt_ne==0)].index.values
C_ = cnt_ne[(cnt_ne>=8)&(cnt_ins==0)&(cnt_ma>=20)].index.values
print(len(A), len(B_), len(C_))
selT = list(rng.choice(A, 40, replace=False)) + list(rng.choice(B_, 30, replace=False)) + list(rng.choice(C_, 30, replace=False))
selT = [int(s) for s in selT]
d = json.load(open('/work/sel.json')); d['selT'] = selT; json.dump(d, open('/work/sel.json','w'))
print(json.dumps(selT))
```