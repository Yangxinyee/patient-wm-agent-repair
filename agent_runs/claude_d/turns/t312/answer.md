Run 282 closes the u-space route as well. u(1.0) for the top NE group is already about 101° away from the low-dose group on the layer-norm sphere. Pushing further along the low→high contrast gives at most about 7% more contrast (gain 1.02–1.075). Getting even that requires `mlp1` outputs far off-manifold, with residuals of 2–19.

Together with run 279 this shows a structural limit. The action encoder cannot express "more NE than the cap" in any direction the downstream model already reads. A dedicated new pathway (encoder → recurrence → head) would be needed. That would require a recurrence Jacobian I can't estimate with the remaining budget, and earlier probes found such a footprint's signal-to-noise around 1 (too costly in factual NLL). I'll keep E5, whose small gain (+0.32) is free on factual records, and accept that ne_x2 stays well below target. Rates above 1.0 never occur in the records, so nothing in the records shows how the model should respond there.

The remaining runs go to verification, which the judging depends on:
1. Factual NLL of the current edit on all 400 tuning stays (the 0.005 budget). This needs the ids in chunks of 100.
2. The four readouts on a second set of stays (sel2), to check that the sel1 calibration generalises. I want the target-computing helpers ready.
3. If the factual budget has room, possibly a small increase of the NE head-channel gain, C, from 2 to about 2.3. That would bring ne_x1/2 from −2.16 to about −2.43.

This run builds the id chunks and inspects the dose_edit CSV format and the target helpers.

```python
import json, pandas as pd
tu = pd.read_csv('/data/tuning.csv', usecols=['stay_id']); allt = sorted(int(x) for x in tu.stay_id.unique())
print(len(allt), 'contiguous' if allt == list(range(2020000000, 2020000400)) else 'NOT contiguous')
s2 = json.load(open('/work/sel2.json')); s2 = s2 if isinstance(s2, list) else (s2.get('stays') or list(s2.values())[0]); s2 = [int(x) for x in s2]
S1 = set(int(x) for x in ids1); print('sel2 n', len(s2), 'in tuning', sum(x in set(allt) for x in s2), 'overlap sel1', sum(x in S1 for x in s2))
rest = [x for x in allt if x not in S1 and x not in set(s2)]; C3, C4 = rest[:100], rest[100:200]
json.dump({'sel1': sorted(S1), 'sel2': s2, 'C3': C3, 'C4': C4}, open('/work/chunks.json','w'))
print('S2', json.dumps(s2)); print('C3', json.dumps(C3)); print('C4', json.dumps(C4))
d13 = pd.read_csv('/work/fwd_13/dose_edit_edited.csv'); print(d13.shape, list(d13.columns)); print(d13.head(3).to_string())
for f_n in [tgt, tgt_rows, true_sh, true_shift, true_shift_ins]:
    co = f_n.__code__; print(f_n.__name__, co.co_varnames[:co.co_argcount], co.co_names[:10])
```