Pooled over all 400 tuning stays (passes 11 to 14, four separate sets of 100 stays, same edit hash), the factual cost of the final edit is +0.00037 nats per reading, with standard error 0.00065. That is far below the 0.005 limit.

| Reading | Change per reading (nats) |
|---|---|
| glucose_finger | +0.0002 |
| glucose_lab | +0.0051 |
| map_arterial | +0.0024 |
| map_cuff | −0.0024 |

The glucose_lab cost is leakage from the insulin readout, and the map_arterial cost comes from the norepinephrine readout. Their combined weight in the overall per-reading average is small.

A bootstrap over stays gives a 99th-percentile cost of 0.0034 for 100 stays and 0.0019 for 400. The chance of a 100-stay set exceeding 0.005 is 0.02%.

So the factual constraint holds with a large margin, and I keep the current edit. I will not push the norepinephrine x1/2 term further. The gain would be about 0.1 to 0.3 of the 0.95 gap, and with only 6 Python runs left a bad rebuild could not be reliably fixed.

**Correcting my previous run:** in run 294 I named a print helper `P`. My notes say `P` was one of the original weight dicts, so I restore it from `PO` here.

**Next step:** a last check of the actual submission's dose readouts. The readouts in passes 7 to 10 were measured before th2 went from 0.83 to 0.868. In this run I will:
- pool those four batches again;
- get the stay lists of the four tuning batches;
- save a backup of the edited tensors.

Then I will run the dose_edit pass with "both" on the four batches and pool the results by hand from the summaries.

**What would confirm the edit:** pooled readouts near insulin x2 ≈ −13.5, insulin x1/2 ≈ 6.5, norepinephrine x2 ≈ 4.45 and norepinephrine x1/2 ≈ −1.3. Clearly different numbers would mean the saved edit is not the configuration I think it is.

```python
import os, json
import numpy as np, pandas as pd
try:
    if isinstance(globals().get('PO'), dict):
        P = PO   # undo clobbering of global P by run 294's print helper
        print('restored P = PO')
except Exception as ex:
    print('restore err', ex)

batches = {}
for k in range(11, 15):
    df = pd.read_csv(f'/work/fwd_{k}/factual_original.csv')
    batches[k] = sorted(int(s) for s in df['stay_id'].unique())

pooled = {}
try:
    for k in range(7, 11):
        js = json.load(open(f'/work/fwd_{k}/summary.json'))
        df = pd.read_csv(f'/work/fwd_{k}/dose_edit_edited.csv')
        if k == 7:
            print('cols', list(df.columns)); print(df.head(3).to_string())
        sc = [c for c in df.columns if 'stay' in c.lower()][0]
        st = set(int(s) for s in df[sc].unique())
        match = [kk for kk, b in batches.items() if st <= set(b)]
        eds = js['results']['edited']['edits']
        print(k, 'csv stays', len(st), 'subset of factual batch', match,
              {e: (v['mean_shift'], v['n_stay_hours'], v['n_stays']) for e, v in eds.items()})
        for e, v in eds.items():
            s0, n0 = pooled.get(e, (0.0, 0))
            pooled[e] = (s0 + v['mean_shift'] * v['n_stay_hours'], n0 + v['n_stay_hours'])
    print('pooled fwd7-10 (th2 .83):', {e: round(s0 / n0, 3) for e, (s0, n0) in pooled.items()},
          {e: n0 for e, (s0, n0) in pooled.items()})
except Exception as ex:
    import traceback; traceback.print_exc()

json.dump({str(k): v for k, v in batches.items()}, open('/work/batches.json', 'w'))
for k, b in batches.items():
    print(f'B{k}', json.dumps(b))

try:
    np.savez('/work/final_edit_backup.npz', **{n: load(n) for n in edited()})
    print('backup saved', len(edited()))
except Exception as ex:
    print('backup err', ex)

with open('/work/notes.md', 'a') as fh:
    fh.write('- Run295: pooled fwd7-10 dose_edit (th2 .83): ' +
             str({e: round(s0 / n0, 3) for e, (s0, n0) in pooled.items()}) +
             '; backup of FINAL edit tensors in /work/final_edit_backup.npz; tuning batches (fwd11-14 stays) in /work/batches.json. '
             'Restored P=PO (run294 clobbered P). Next: dose_edit both on 4 batches as final check, then FINAL ANSWER.\n')
print('notes appended')
```