The truth on sel1 is a bit smaller than my notebook said. Recomputed on the tool's own reached rows, it is −13.25 for insulin ×2 and +6.63 for ×½ (norepinephrine: +4.26 and −2.13). So the edited model's −12.13 and +5.67 are 92% and 86% of the truth.

Two other findings from that run:
- **Offline estimate:** linear interpolation between f = 1.2 and 1.35 puts f = 1 at about −12.40 and +5.83 on these 100 stays. That makes the offline copy about 2.5% stronger than the tool.
- **Mismatch in timing and dose dependence:** the edited model's effect grows steadily by about 1.9 per hour. The true effect levels off near −15 by hour 18. The edited model also depends more steeply on glucose level than the truth does. Because of this, how well it matches depends on which stays are in the sample, which explains the 40 versus 60 split.

The score is a pooled mean over unseen stays, so the robust fix is to calibrate f on a larger sample. The replica is fast (about 0.28 s per stay), so this run rolls out all 400 tuning stays at f = 1.1 and 1.2. I will pick f so the offline value is about 1.025 × truth.

In the same run I prepare the norepinephrine work, given the 14 runs left after this one:
- the truth's time course for norepinephrine on the sel1 rows;
- the lib3 source for capturing model states and editing doses;
- the dataset objects in the kernel.

My idea for norepinephrine is to reuse the original model's own encoding of the dose in its state, rather than building a new memory. I would add a readout gated to map_arterial along the most cost-efficient state direction.

What would confirm or refute the current plan:
- **Confirm:** the 400-stay ratio lands near 0.9, so f ≈ 1.10–1.15.
- **Refute:** a very different ratio would mean even 100 stays were not representative.

```python
import time, inspect, json, numpy as np, pandas as pd
T0=time.time()
ids = sorted(tu.stay_id.unique().tolist()); print('tuning stays', len(ids))
resF = {}
for f in [1.1, 1.2]:
    t0=time.time(); R = eval_shifts(scaledP(f), ids, 20, 0); resF[f]=R
    print('f', f, {k:(round(float(v[0]),2), round(float(v[1]),2), v[2]) for k,v in R.items()}, 'sec', round(time.time()-t0))
# NE truth hour profile on fwd5 reached rows
try:
    de = pd.read_csv('/work/fwd_5/dose_edit_edited.csv')
    for ed in ['ne_x2','ne_half']:
        q_ = de[de.reached & (de.edit==ed)].copy()
        q_['truth'] = [true_shift(int(s), int(t), ed) for s, t in zip(q_.stay_id, q_.hour)]
        gh = q_.groupby('hour')[['shift','truth']].mean().round(1)
        print(ed, 'pooled ed', round(q_['shift'].mean(),2), 'tr', round(q_.truth.mean(),2), 'hours', gh.index.tolist(), 'ed', gh['shift'].tolist(), 'tr', gh['truth'].tolist())
except Exception as ex: print('ne diag err', repr(ex)[:300])
print(open('/work/lib3.py').read())
gl = [k for k,v in globals().items() if not k.startswith('_') and (k.startswith(('ds','a0','WC','N3','base','nll','sel','age','tr','tu','PB','PA','bet','b0g','gdir','tilt','theta','th')))]
print('globals', gl)
print('total sec', round(time.time()-T0))
```