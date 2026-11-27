My bundle construction reproduces the tool's atom counts for the first 12 bundles of stay 2020000000. Before building on it, I'll check that bundle times and atom counts match the tool for all 100 stays. Those 5,116 bundles are the units against which every later state comparison is made, so any mismatch (for example in END-atom placement or rounding of times) has to be found now.

```python
bad = 0; tot = 0; Bs = {}
for sid in sel1:
    Bk = stay_bundles(tu_g[sid]); Bs[sid] = Bk
    m = S2['stay_id']==sid
    th = S2['hour'][m]; na = S2['n_atoms'][m]
    mine_t = np.array([t for t,_ in Bk]); mine_n = np.array([len(a) for _,a in Bk])
    ok = len(th)==len(mine_t) and np.allclose(th, mine_t, atol=1e-3) and (na==mine_n).all()
    bad += (not ok); tot += 1
    if not ok and bad<3: print(sid, len(th), len(mine_t), th[:8], mine_t[:8], na[:8], mine_n[:8])
print('mismatched stays:', bad, 'of', tot)
```