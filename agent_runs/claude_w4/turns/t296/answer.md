fwd_8 turns out to be the unnoted test of the v4 edit: a dose_edit "both" run on 100 tuning stays. The output was cut before the edited-model summary, but two things already show:

- **The edit is active.** The spread of per-stay-hour shifts rose from 0.53 (original) to 8.6 (edited).
- **It also moves the recorded-dose rollout.** The predicted level under recorded doses rose by about 9 on average, and by up to +50 mg/dL in one stay. This fits a known flaw of v4: it centres the insulin signal on a fixed rate (1.77) instead of on the model's own expected dose. So at low glucose bands it pushes glucose up every hour, at high bands it pushes it down, and the push builds up over the rollout. That would hurt factual fit, and it would make "stop the drug" (x0) wrong if a stopped drug means no infusion record.

Next I will print the edited summary, the per-hour shape of the shifts against the true response curve, and the drift by hour. I will also check whether fwd_8 used the sel100 stays and whether a truth function survives in the kernel.

- If the mean shifts are near truth (−12.96 / +4.94 on sel100), the gains are roughly right.
- If the drift grows with hour, I must fix the centering before tuning the gains.

```python
import json, numpy as np, pandas as pd
S8 = json.load(open('/work/fwd_8/summary.json'))
for m in ['original','edited']:
    print(m, {k:(v['mean_shift'],v['n_stay_hours'],v['n_stays']) for k,v in S8['results'][m]['edits'].items()})
print('other summary keys:', [k for k in S8.keys() if k!='results'], S8.get('stays', None) is not None)
de = pd.read_csv('/work/fwd_8/dose_edit_edited.csv'); do = pd.read_csv('/work/fwd_8/dose_edit_original.csv')
st8 = sorted(de.stay_id.unique()); print('n stays', len(st8), 'same as sel100:', set(st8)==set(sel100) if 'sel100' in globals() else 'no sel100')
print('in sel100:', len(set(st8)&set(sel100)) if 'sel100' in globals() else None)
for ed in ['insulin_x2','insulin_half','ne_x2','ne_half']:
    a = de[(de.edit==ed)&de.reached]; b = do[(do.edit==ed)&do.reached]
    print(ed, 'per-hour mean shift edited:', a.groupby('hour').shift.mean().round(2).to_dict())
    print('    orig:', b.groupby('hour').shift.mean().round(2).to_dict())
# factual drift (edited minus original rollout under recorded doses), by edit target and hour
m = de.merge(do, on=['stay_id','hour','edit'], suffixes=('_e','_o'))
m['drift'] = m.factual_e - m.factual_o
for ed in ['insulin_x2','ne_x2']:
    mm = m[m.edit==ed]
    print(ed, 'factual drift by hour: mean', mm.groupby('hour').drift.mean().round(1).to_dict())
    print('   abs', mm.groupby('hour').drift.apply(lambda x: np.abs(x).mean()).round(1).to_dict())
print('truth-ish funcs in kernel:', [k for k in globals() if any(s in k.lower() for s in ['truth','irf','true_','htrue','hf','ha'])][:40])
```