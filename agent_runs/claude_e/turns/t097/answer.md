With my explicit regression specification, the model's coefficients on insulin dose deviation (lags 1–6) are [−1.45, +0.39, +0.18, −0.89, −0.02, −0.65], with standard errors around 0.33. The earlier specification gave [−0.34, +0.13, ...]. The true coefficients agree across both specifications: [−2.04, −1.23, −0.25, +0.38, +0.80, +0.69].

So the model's own dose response is too noisy, and too sensitive to the specification, to subtract reliably from 200 stays. The two estimates of the per-lag gap Δβ disagree even in sign at later lags. Glucose-persistence terms broadly agree between model and truth, except the lab terms, where the model is weaker.

Next step: get many more model states, from train stays that are rich in both drugs after hour 12. These give precise estimates of the model's own dose coefficients, using the local head to get predicted means. They also give enough rows to fit a dose read-out from the head input.

I'll select about 600 such train stays now and request their states in batches. If the model's lag-1 coefficient converges somewhere between −0.3 and −1.5 with a standard error near 0.1, the correction plan is supported. If it varies with specification even with lots of data, the model's dose response is nonlinear, and I should calibrate against the forward tool's dose-edit outputs instead.

```python
import json
ins_after = Gtr['ins'].loc[:, 12:24].notna().sum(1)
ne_after = Gtr['ne'].loc[:, 12:24].notna().sum(1)
ma_cnt = Gtr['ma'].loc[:, 12:24].notna().sum(1)
cand_both = [int(s) for s in Gtr['gf'].index[(ins_after >= 3) & (ne_after >= 3) & (ma_cnt >= 6)]]
cand_ins = [int(s) for s in Gtr['gf'].index[(ins_after >= 4)]]
print('cand both drugs + art line:', len(cand_both), ' cand insulin>=4h:', len(cand_ins))
rng = np.random.default_rng(7)
selA = list(rng.choice(cand_both, size=400, replace=False))
rest = [s for s in cand_ins if s not in set(selA)]
selB = list(rng.choice(rest, size=200, replace=False))
batches = [sorted(int(x) for x in selA[i:i+100]) for i in range(0, 400, 100)] + [sorted(int(x) for x in selB[i:i+100]) for i in range(0, 200, 100)]
json.dump(batches, open('/work/sel_tr2.json', 'w'))
print([len(b) for b in batches])
print(json.dumps(batches[0]))
```