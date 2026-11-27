The table maps each factual pass to its batch and edit. The usable insulin-only measurements are:

| Batch | Pass | Gain | Cost |
|---|---|---|---|
| f3 | fwd_14 | -2.5 | .00283 |
| tu0 | fwd_13 / fwd_26 / fwd_25 | -2.6 / -2.7 / -2.8 | .00428 / .00459 / .00492 |
| tu1 | fwd_15 / fwd_29 | -2.5 / -2.7 | .00369 / .00428 |
| tu2 | fwd_17 | -2.6 | .00100 |
| four training batches | fwd_18–21 | -2.6 | .00164, .00261, .00368, .00104 |

Gain² scaling reproduces the measured tu0 costs at -2.7 and -2.8 to within .00004, so I will use it to put every stay on a common gain.

Now the stay-level bootstrap. Each stay contributes its summed NLL difference and its number of readings, scaled by (a/a_measured)². For evaluation sets of N stays I compute the probability that the pooled cost exceeds .005, using all 800 stays and also tuning stays only.

```python
src = {'tu0': (26, 2.7), 'tu1': (29, 2.7), 'f3': (14, 2.5), 'tu2': (17, 2.6), 'tr0': (18, 2.6), 'tr1': (19, 2.6), 'tr2': (20, 2.6), 'tr3': (21, 2.6)}
ps = []
for lab, (k, g) in src.items():
    a = pd.read_csv(f'/work/fwd_{k}/factual_original.csv'); b = pd.read_csv(f'/work/fwd_{k}/factual_edited.csv')
    a['d'] = b['nll'].values - a['nll'].values
    t = a.groupby('stay_id')['d'].agg(['sum','count']).reset_index(); t['lab'] = lab; t['g'] = g; ps.append(t)
PS = pd.concat(ps, ignore_index=True); PS['tun'] = ~PS.lab.str.startswith('tr')
rng = np.random.default_rng(0)
res = []
for pool in ['all', 'tuning']:
    Q = PS if pool=='all' else PS[PS.tun]
    for gain in [2.5, 2.6, 2.65, 2.7]:
        cs = (Q['sum'] * (gain/Q['g'])**2).values; ns = Q['count'].values
        row = [pool, gain, round(cs.sum()/ns.sum(), 5)]
        for N in [100, 200, 400, 800]:
            idx = rng.integers(0, len(cs), size=(20000, N))
            m = cs[idx].sum(1) / ns[idx].sum(1); row.append(round((m > 0.005).mean()*100, 2))
        res.append(row)
print(pd.DataFrame(res, columns=['pool','gain','mean','P%N100','P%N200','P%N400','P%N800']).to_string())
```