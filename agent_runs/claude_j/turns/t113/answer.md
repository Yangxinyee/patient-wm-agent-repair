Several hidden units of the value head are effectively dead. Units 89, 170 and 149 have a largest contribution of 0.003 to 0.014 logits and pre-activations that never exceed −2.2. I can therefore repurpose 89 for an insulin correction (gated to glucose codes) and 170 for a norepinephrine correction (gated to MAP codes), with a negligible change to the current function.

Each correction will be a state readout. I will fit it by regression, with the target being the history of dose minus expected dose, weighted by the missing part of the true kernel. To fit it well I need more states than the 100 tuning stays provide. I will choose 300 training stays that are rich in insulin and norepinephrine and fetch their states in three requests. First I record the findings and write the stay selection.

```python
add4 = """
- HEAD REPLICA (/work/heads.py, runs 99-103): value logits = value_bin.2(gelu(value_bin.0(with_code))) + embed_transition[code*17+lastbin];
  q = query_norm(view + type_emb[1]); view = out + out_norm(mean_slots(h*exp(-dt*exp(flow.a)))) + gap(log(dt_s)/10) + z_head(z), z~prior;
  last_numeric cols: 0 = has_last, 3 = log1p(age_s)/10, 6 = z_last_value (others 0 for measurements); last_status NOT added; bins[0] rows zero.
  MAE vs tool predicted mean: 0.94 (readings with a last value), 2.3 on first readings (only first GF after admission is off).
- Dead hidden units in value_bin.0: 89, 170, 149 (max contribution <0.015 logits). PLAN: repurpose 89 (insulin->GF/GL) and 170 (NE->MA/MC),
  gated by embed.code, linear regime, constant offset cancelled in embed_transition rows; readout fit by ridge on q.
"""
with open('/work/notes.md', 'a') as fh:
    fh.write(add4)
print(tr.columns.tolist(), tr.stay_id.nunique())
ins_hr = tr[(tr.event == 'insulin') & (tr.hour >= 6) & (tr.hour < 25)].groupby('stay_id').size()
ne_hr = tr[(tr.event == 'norepinephrine') & (tr.hour >= 6) & (tr.hour < 25)].groupby('stay_id').size()
print('stays with >=4 insulin rows h6-25:', int((ins_hr >= 4).sum()), ' with >=4 NE rows:', int((ne_hr >= 4).sum()))
rng_ = np.random.default_rng(7)
poolI = ins_hr[ins_hr >= 4].index.values; poolN = ne_hr[ne_hr >= 4].index.values
selT = list(rng_.choice(poolI, 150, replace=False)) + list(rng_.choice(np.setdiff1d(poolN, poolI), 150, replace=False))
selT = [int(s) for s in selT]
rng_.shuffle(selT)
json.dump(selT, open('/work/sel_train.json', 'w'))
print('selected', len(selT), 'first', selT[:3])
```