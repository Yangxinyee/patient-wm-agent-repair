The states came back: 5,116 bundles for 100 stays, about 51 per stay. To rebuild bundles locally I must reproduce exactly which events form a bundle and how many atoms each holds. In particular, I need to know whether infusion END atoms (at s + 1.083 h) sit in the same bundle as the next record's start, and whether hospital and ICU admission atoms are counted.

I'll load the npz and compare its bundle list (hours, n_atoms, drug flags) for one stay against that stay's raw rows. A rule of "one bundle per distinct minute, with END atoms at end_hour" should reproduce the atom counts exactly.

```python
S2 = np.load('/work/fwd_2/states_original.npz'); print({k: S2[k].shape for k in S2.files})
sid0 = 2020000000
m = S2['stay_id']==sid0
print(pd.DataFrame({'hour':S2['hour'][m],'ins':S2['has_insulin'][m],'ne':S2['has_norepinephrine'][m],'n_atoms':S2['n_atoms'][m]}).T.to_string())
print(tu[tu.stay_id==sid0][['hour','event','value','rate','dose','end_hour']].to_string())
```