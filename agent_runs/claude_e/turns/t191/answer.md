The previous output was cut off. What it did show:
- `fwd_10` is a dose_edit pass with "both" models on 100 tuning stays (they begin at 2020000000, so probably the f3 set).
- `fwd_11` is a factual pass with "both".
- The kernel keeps no input history, so I have to rebuild what happened from the files.

I'll print only the key numbers from the two summary.json files: the mean shift per edit for original and edited, and the NLL figures.

If the factual delta is near or above 0.005, the current saved edit is too costly. If the insulin ×2 shift is far from about −13.8, the gains need recalibrating.

```python
import json
def flat(d, pre=''):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, pre + k + '.'))
    elif isinstance(d, list):
        out[pre[:-1]] = str(d)[:200]
    else:
        out[pre[:-1]] = d
    return out
for k in [10, 11]:
    s = json.load(open(f'/work/fwd_{k}/summary.json'))
    fl = flat(s)
    print('--- fwd', k, 'n keys', len(fl))
    for kk, vv in fl.items():
        if isinstance(vv, str) and len(vv) > 120:
            vv = vv[:120]
        if 'what' in kk:
            continue
        print(kk, ':', vv)
```