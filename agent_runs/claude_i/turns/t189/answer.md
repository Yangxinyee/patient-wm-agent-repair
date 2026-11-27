Layout (has-last in column 1, log-age in column 3, z-last-value in column 2, status 0) makes the head match the tool on MAP: median error 0.97 for arterial and 1.21 for cuff. Glucose is still off: 7.2 for finger and 3.9 for lab. So the shared head is right and something glucose-specific is missing.

One candidate is that glucose's register uses the last insulin rate in its rate fields. Another is that finger and lab share one glucose register. I'll check whether the finger-glucose residual tracks the last insulin rate or the time since insulin. A strong correlation with insulin favours the cross-code rate register. If there is none, I'll test a shared glucose register next.

```python
m_ = vm_custom(ctxS, codesS, lvS, laS/3600, best_pos, best_st)
FS = Fm.iloc[sub_i].copy(); FS['res'] = m_ - tgtS
ins_tr = train[(train.event=='insulin') & train.stay_id.isin(set(S3.tolist()))][['stay_id','hour','rate']]
def last_ins(s, h):
    d = ins_tr[(ins_tr.stay_id==s)&(ins_tr.hour<h)]
    return (d.rate.iloc[-1], h-d.hour.iloc[-1]) if len(d) else (np.nan, np.nan)
li = np.array([last_ins(s,h) for s,h in zip(FS.stay_id, FS.hour)])
FS['lir'] = li[:,0]; FS['lia'] = li[:,1]
for ev_ in ['glucose_finger','glucose_lab']:
    w_ = FS.event==ev_
    d = FS[w_]
    print(ev_, 'res mean', d.res.mean().round(2), 'sd', d.res.std().round(2),
          '| corr w lastins rate', np.corrcoef(np.nan_to_num(d.lir), d.res)[0,1].round(3),
          '| res when no ins', d.res[d.lir.isna()].abs().median().round(2), 'with ins', d.res[d.lir.notna()].abs().median().round(2))
```