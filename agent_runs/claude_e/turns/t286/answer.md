The kernel ran out of memory (it was already holding large arrays from earlier runs) and restarted, so all variables and helper functions are gone. Saved edits are stored on disk through `save_edit`, so the submitted edit should still be `ins_only(-2.7)`. I need to verify that first.

I can no longer call `ins_only`. If needed, I can still derive other gains: the gain enters only through W2's output columns for the readout units 26/41, and should be proportional to the gain because repurposing those units costs nothing at gain 0.

In this run I will:
1. Check which tensors are edited and which rows/columns differ from the original. The expected pattern is W0 rows 26/41, b0 entries 26/41 and W2 columns 26/41 only.
2. Redo the stay-level cost bootstrap memory-lightly, in chunks.

```python
import json, os, numpy as np, pandas as pd, torch
print('edited:', edited())
for n in edited():
    a = load(n).numpy(); b = load_original(n).numpy(); d = np.abs(a-b)
    if d.ndim == 2: print(n, a.shape, 'rows', np.where(d.max(1)>1e-7)[0][:10], 'cols', np.where(d.max(0)>1e-7)[0][:10])
    else: print(n, a.shape, 'idx', np.where(d>1e-7)[0][:10])
W2o = load_original('attr_heads.value_bin.2.weight').numpy(); W2e = load('attr_heads.value_bin.2.weight').numpy()
print('W2 cols 26/41 orig norms', np.linalg.norm(W2o[:, [26,41]], axis=0), 'edited norms', np.linalg.norm(W2e[:, [26,41]], axis=0))
print('edited col26 vs col41 cos', float(W2e[:,26]@W2e[:,41]/np.linalg.norm(W2e[:,26])/np.linalg.norm(W2e[:,41])))
src = {'tu0': (26, 2.7), 'tu1': (29, 2.7), 'f3': (14, 2.5), 'tu2': (17, 2.6), 'tr0': (18, 2.6), 'tr1': (19, 2.6), 'tr2': (20, 2.6), 'tr3': (21, 2.6)}
ps = []
for lab, (k, g) in src.items():
    a = pd.read_csv(f'/work/fwd_{k}/factual_original.csv', usecols=['stay_id','nll']); b = pd.read_csv(f'/work/fwd_{k}/factual_edited.csv', usecols=['nll'])
    a['d'] = b['nll'].values - a['nll'].values
    t = a.groupby('stay_id')['d'].agg(['sum','count']).reset_index(); t['lab'] = lab; t['g'] = g; ps.append(t)
PS = pd.concat(ps, ignore_index=True); PS['tun'] = ~PS.lab.str.startswith('tr'); PS.to_pickle('/work/stay_costs.pkl')
rng = np.random.default_rng(0); res = []
for pool in ['all', 'tuning']:
    Q = PS if pool=='all' else PS[PS.tun]
    for gain in [2.5, 2.6, 2.65, 2.7]:
        cs = (Q['sum'] * (gain/Q['g'])**2).values; ns = Q['count'].values.astype(float)
        row = [pool, gain, round(cs.sum()/ns.sum(), 5)]
        for N in [100, 200, 400, 800]:
            hits = 0
            for _ in range(5):
                idx = rng.integers(0, len(cs), size=(2000, N), dtype=np.int32)
                hits += ((cs[idx].sum(1) / ns[idx].sum(1)) > 0.005).sum()
            row.append(round(hits/10000*100, 2))
        res.append(row)
print(pd.DataFrame(res, columns=['pool','gain','mean','P%N100','P%N200','P%N400','P%N800']).to_string())
```