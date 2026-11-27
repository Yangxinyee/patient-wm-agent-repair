The tool confirms the insulin repair, though at a somewhat smaller size than offline. On all 100 sel1 stays, the edited model gives:
- insulin ×2: −12.13, against my truth estimate of −14.30, about 85% of it;
- insulin ×½: +5.67, against +7.15, about 79%.

The original model gives −1.77 and +0.86, which matches what fwd1 showed. Offline, I had matched the truth almost exactly (−14.93 against −14.98), but only on 40 of these 100 stays.

There are two possible reasons for the gap, and they lead to different fixes:
- **(a) Different stays:** those 40 stays respond more strongly than the other 60. Then I just recalibrate theta on all 100 stays; my rough linear estimate is about −2.9.
- **(b) Tool and replica disagree:** the edited model behaves differently under the tool's rollout than in my offline replica. Then I must calibrate against the tool itself.

To tell them apart, I split the tool's per-stay-hour shifts into the first 40 sel1 stays (assumed to be the calibration set) and the other 60.
- **(a) is confirmed if** the tool gives about −14.9 on those 40.
- **(b) is indicated if** the tool gives about −12 on those 40 as well.

In the same run I also try to recover the source code of my kernel-only functions. If the source is lost, I list their hard-coded constants instead. This tells me whether I can reuse the same readout machinery for norepinephrine.

```python
import pandas as pd, numpy as np, inspect, json
de = pd.read_csv('/work/fwd_5/dose_edit_edited.csv'); do = pd.read_csv('/work/fwd_5/dose_edit_original.csv')
print(de.shape, list(de.columns)); print(de.head(2).to_string())
s40 = set(sel['sel1'][:40])
try:
    sc = [c for c in de.columns if 'stay' in c.lower()][0]
    for nm, d in [('edited', de), ('orig', do)]:
        if 'edit' in d.columns:
            shc = [c for c in d.columns if 'shift' in c.lower()][0]
            for ed, g in d.groupby('edit'):
                gr = g[g['reached'].astype(bool)] if 'reached' in g.columns else g
                i4 = gr[sc].isin(s40)
                print(nm, ed, '40:', round(gr[i4][shc].mean(),2), len(gr[i4]), 'rest60:', round(gr[~i4][shc].mean(),2), len(gr[~i4]))
        else:
            print('wide format; numeric cols:', [c for c in d.columns if d[c].dtype.kind in 'fib'])
except Exception as ex:
    print('ERR', repr(ex))
for f in [eval_shifts, add_readout, plumb, true_shift, rollout_stay]:
    try: print(f.__name__, str(inspect.signature(f)))
    except Exception as ex: print('sig err', ex)
for f in [eval_shifts, add_readout, plumb]:
    try:
        src = inspect.getsource(f); print('SRC', f.__name__, len(src)); print(src[:1800])
    except Exception as ex:
        c = f.__code__
        print('NOSRC', f.__name__, 'consts', [x for x in c.co_consts if not hasattr(x,'co_code')][:60], 'names', c.co_names[:80])
```