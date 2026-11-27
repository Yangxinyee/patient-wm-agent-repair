The pass succeeded, but I made another error. I did not print the second batch of my 300 selected ids before writing the request, so I typed ids that were not in that list. They are valid training stays, since the tool returned 100 of them. With 4,247 bundles against 5,380 for the selected batch, they look less drug-heavy, but they are still usable probe data.

From now on I will print ids before every request. This run:
1. adds these states to the probe set and re-checks the probe R² on all 299 stays;
2. prints the next 100 selected ids exactly, so the next request uses real ids.

If R² keeps rising, I will add the remaining selected stays as well.

```python
S6 = np.load('/work/fwd_6/states_original.npz')
D6 = build_set(S6['x'], S6['stay_id'], S6['hour'])
s6 = sorted(set(S6['stay_id'].tolist())); print('fwd6 stays', len(s6), 'in new300:', len(set(s6) & set(new300)), 'insulin hours mean', np.mean([nins[sidx[s_]] for s_ in s6]), 'ne', np.mean([nne[sidx[s_]] for s_ in s6]))
for nm_, combo in [('ins eps1', [1, 0, 0]), ('ins -1.5e1-1.4e2', [-1.5, -1.4, 0])]:
    print(nm_, '299 stays', probe_r2([D99, D5, D6], GL, eps_ins, np.array(combo)))
print('ne eps1 299', probe_r2([D99, D5, D6], MP, eps_ne, np.array([1, 0, 0])))
rest = [s_ for s_ in new300 if s_ not in set(S5['stay_id'].tolist())]
print(len(rest)); print(json.dumps(rest[:100]))
```