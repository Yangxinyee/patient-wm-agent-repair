The states pass returned 644 bundles for the 12 stays, about 54 per stay. Each row has out, the 4 slots of h and the z posterior mean, plus the hour and atom count.

Next I will match these bundles to the raw records for one stay. I want to learn how events form bundles and atoms: whether infusion END atoms share the minute with the next record's start, and whether each infusion record counts as one atom or two. If n_atoms at the drug bundles is 2 when one record ends as the next begins, then END atoms are separate atoms in the same bundle.

```python
F4 = np.load('/work/fwd_4/states_original.npz')
print({k_: F4[k_].shape for k_ in F4.files})
s0 = 2020000003
mk = F4['stay_id']==s0
print(np.round(F4['hour'][mk][:30],3).tolist())
print(F4['n_atoms'][mk][:30].tolist())
print(F4['has_insulin'][mk][:30].astype(int).tolist(), F4['has_norepinephrine'][mk][:30].astype(int).tolist())
r0 = tu[tu.stay_id==s0].sort_values('hour')
print(r0[['hour','event','value','rate','end_hour']].head(25).round(3).to_string(index=False)[:1800])
```