The states file holds 5,301 bundles from 100 stays, with 992 numbers per bundle in the order out, 4 slots of h, z mean. It also gives the bundle hour, the drug flags and n_atoms for each bundle. n_atoms tells me exactly how the model grouped events into bundles and whether END atoms count. So I will compare one stay's bundle times and atom counts with what my build_bundles produces. If they disagree, for example extra END atoms or a different grouping, that is a concrete bug to fix before anything else. If they agree, the bug is in the arithmetic.

```python
sid0 = 1010001088
ev0 = tr[tr.stay_id==sid0].sort_values('hour')
print(ev0.head(14).to_string())
m0 = S_['stay_id']==sid0
print(list(zip(np.round(S_['hour'][m0][:16],3), S_['n_atoms'][m0][:16], S_['has_insulin'][m0][:16].astype(int), S_['has_norepinephrine'][m0][:16].astype(int))))
opts0 = dict(unit={e:0 for e in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']}, ru={'insulin':0,'norepinephrine':0}, end_attrs=False)
bb0 = build_bundles(ev0, opts0)
print([(round(t,3), len(a)) for t,a in bb0[:16]], len(bb0), m0.sum())
```