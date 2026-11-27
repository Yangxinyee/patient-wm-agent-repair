`gated3(-1.9,3)` is identical to the saved edit, so the globals did not change. The offline evaluation of that exact edit on sel1 still shows map_arterial rows blowing up (dNLL +2.05, mean shifts with sd 10.6). The tool on the f3 tuning set showed only +0.0053 per map_arterial reading.

So either the blow-up is real for some stays and the evaluation set could hit it too, or my offline sel1 pipeline is wrong for these rows. Separately, zeroing the 4 repurposed units already costs 0.0013 offline.

The 38 stays in both sel1 and f3 decide it. I'll match rows between the offline sel1 output and the tool's per-reading factual output (fwd_11) and compare predicted-mean shifts and NLL changes on map_arterial and glucose_finger. I'll also list which stays carry the large offline shifts.

```python
RRc['stay'] = RRc['stay'].astype(np.int64)
tool = fe[['stay_id','hour','event','predicted_mean','nll']].copy()
tool['dpm_t'] = fe['predicted_mean'].values - fo['predicted_mean'].values
tool['dn_t'] = fe['nll'].values - fo['nll'].values
assert (fe.stay_id.values==fo.stay_id.values).all()
tool['hr4'] = tool['hour'].round(3); RRc['hr4'] = RRc['hour'].astype(float).round(3)
print('RR events', RRc.event.unique()[:6], 'tool events', tool.event.unique()[:6])
mm_ = RRc.merge(tool, left_on=['stay','event','hr4'], right_on=['stay_id','event','hr4'], how='inner')
print('merged rows', len(mm_), 'stays', mm_.stay.nunique())
for ev in mm_.event.unique():
    s_ = mm_[mm_.event==ev]
    print(ev, 'n', len(s_), 'offline dpm sd %.3f dn %.4f | tool dpm sd %.3f dn %.4f | corr dpm %.3f' % (
        s_.dpm_s.std(), s_.dn_s.mean(), s_.dpm_t.std(), s_.dn_t.mean(), np.corrcoef(s_.dpm_s, s_.dpm_t)[0,1]))
big = RRc[(RRc.code==7) & (RRc.dpm_s.abs()>5)]
print('big ma rows', len(big), 'stays', big.stay.nunique(), 'in f3', len(set(big.stay) & set(map(int,f3_ids))))
print(big.groupby('stay').size().sort_values(ascending=False).head(8).to_dict())
```