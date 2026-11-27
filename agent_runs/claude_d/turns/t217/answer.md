Teacher-forced, the edited head shows no large glucose bias: on insulin-present rows at hour 13 it is -0.7 mg/dL, against +17 in the rollout. So the drift comes from the rollout itself.

The factual-fit cost is also spread badly. The channel damages rows where the drug was absent:
- glucose_finger without prior insulin: +0.012.
- On map_cuff and map_arterial the cost sits mainly on the NE-present rows: +0.028 and +0.015.

So the gates leak and the MAP channel is costly. E2 is not usable as it stands. That holds whatever is causing the drift.

To locate the drift, I will compare predicted means at the first hidden reading. Hour 13 is the first hour of the rollout, and its glucose reading is predicted from an almost real history.
- If the original model's rollout mean at hour 13 matches its teacher-forced mean, the rollout machinery is as I assume.
- If, for the edited model, the rollout mean sits about 17 above its own teacher-forced mean on the same readings, then something drawn in the rollout before that reading is shifting the state. That something would be sampled MAP or lab values, or class-0 ("no value") draws.

```python
tf = m9[(m9.event=='glucose_finger')].copy()
tf1 = tf.groupby(['stay_id','h']).agg(pm_o=('predicted_mean','mean'), pm_e=('predicted_mean_e','mean'), obs=('observed','mean')).reset_index()
ro = d8e[d8e.edit=='insulin_x2'][['stay_id','hour','factual']].rename(columns={'factual':'ro_e'})
ro['ro_o'] = d8o[d8o.edit=='insulin_x2'].factual.values
mg = ro.merge(tf1, left_on=['stay_id','hour'], right_on=['stay_id','h'], how='inner')
for h_ in [13,14,15,18,24]:
    a_ = mg[mg.hour==h_]
    print(h_, 'n', len(a_), 'ro_o-tf_o %.2f  ro_e-tf_e %.2f  ro_o-obs %.1f  tf_o-obs %.1f' % ((a_.ro_o-a_.pm_o).mean(), (a_.ro_e-a_.pm_e).mean(), (a_.ro_o-a_.obs).mean(), (a_.pm_o-a_.obs).mean()))
a_ = mg[mg.hour==13]
print('h13 per-stay ro_e-tf_e pct:', np.round(np.percentile(a_.ro_e-a_.pm_e, [10,50,90]),1), ' ro_o-tf_o pct:', np.round(np.percentile(a_.ro_o-a_.pm_o,[10,50,90]),1))
# bundles before first gf reading in hour 13: which events occur in hour 13 earlier?
first13 = tf[tf.h==13].groupby('stay_id').hour.min()
ev13 = m9[(m9.h==13)]
pre = [((ev13.stay_id==s_) & (ev13.hour < t_)).sum() for s_, t_ in first13.items()]
print('readings in hour13 before first gf: mean %.2f, frac>0 %.2f' % (np.mean(pre), np.mean(np.array(pre)>0)))
```