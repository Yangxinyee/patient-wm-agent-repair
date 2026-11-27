The states file has 5,075 rows for 100 stays, about 51 bundles each. Each row is the state after one bundle: out (192), four h slots (192 each), then the 32-d z posterior mean. Stay, hour, drug flags and atom count are stored alongside.

Before reimplementing anything, I need to know exactly how the model groups records into bundles:
- Is each hour's readings one bundle and each drug minute (h+0.083) another?
- Are infusion END atoms counted, and where?
- How are the admission rows handled?

I'll compare the bundle list for one stay with its records, using the atom counts and times.

```python
z = np.load('/work/fwd_3/states_original.npz')
print({k: z[k].shape for k in z.files})
sid0 = 2020000176
msk = z['stay_id']==sid0
print(pd.DataFrame({'hour':z['hour'][msk], 'ins':z['has_insulin'][msk], 'ne':z['has_norepinephrine'][msk], 'n_atoms':z['n_atoms'][msk]}).to_string())
print(tu[tu.stay_id==sid0].drop(columns=['rate_unit','dose_unit']).head(40).to_string())
```