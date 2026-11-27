The error is mostly a common offset (|mean| 0.076) plus a sizeable spread (0.049), and none of the ctx variants fixes it. So both the drug-atom constants and something per-bundle are off. My earlier self spent many runs on this reconstruction without making it exact, so I won't sink more runs into it yet.

One discrepancy is cheap to resolve first. Right after an action bundle, `out` carries the action-B residual at R² about 0.23. At the next reading, the per-reading `out_` gives only 0.06. If `out_` (and so Q99) was built from the wrong state, for example the state after the reading itself instead of after the preceding action bundle, that would explain both the lost signal and my head reconstruction's error of about 2 units. I'll check which recorded state `out_` matches: the bundle before the reading, or the reading's own bundle. If it's the wrong one, I'll rebuild q from the correct state, and the head-level probe may be much stronger than it looked.

```python
key2j = {(int(s), round(float(h), 4)): j for j, (s, h) in enumerate(zip(sidS, hrS))}
d_prev, d_same, n = [], [], 0
for i in range(0, len(Rr), 7):
    s, h = int(Rr.stay_id[i]), float(Rr.hour[i]); j = key2j.get((s, round(h, 4)))
    if j is None or j == 0 or sidS[j - 1] != s: continue
    d_prev.append(np.abs(out_[i] - xs[j - 1, :192]).max()); d_same.append(np.abs(out_[i] - xs[j, :192]).max()); n += 1
print('n', n, 'out_ vs prev-bundle out: median maxdiff %.2e' % np.median(d_prev), '| vs same-bundle out: %.2e' % np.median(d_same))
print('prev bundle hour frac example', [round(float(hrS[key2j[(int(Rr.stay_id[i]), round(float(Rr.hour[i]), 4))] - 1]), 4) for i in range(30, 36)])
```