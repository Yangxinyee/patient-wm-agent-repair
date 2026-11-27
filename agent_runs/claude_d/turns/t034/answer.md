The states request returned 156 rows for 3 stays, about 52 bundles per stay, and arrived in 2 seconds. Each row holds out, the 4 h slots and the z posterior mean after one bundle, plus the hour, the drug flags and the atom count.

Next I will line up the bundle sequence of stay 2020000000 against its raw events. That shows how the record becomes bundles: whether drug records sit in their own hh:05 bundles, whether END atoms are counted, and whether hospital and ICU admission are separate bundles. I will also look at basic properties of the states, such as norms and whether out looks layer-normed. This is the groundwork for reimplementing the model.

```python
S = np.load('/work/fwd_2/states_original.npz')
print(S.files)
for k in S.files: print(k, S[k].shape, S[k][:3] if S[k].ndim==1 else S[k][:2,:5])
x = S['x'] if 'x' in S.files else None
sid_ = S['stay_id']; hr_ = S['hour']; hi_ = S['has_insulin']; hn_ = S['has_norepinephrine']; na_ = S['n_atoms']
m0 = sid_==2020000000
print(pd.DataFrame({'hour':hr_[m0],'ins':hi_[m0],'ne':hn_[m0],'n':na_[m0]}).T.to_string())
print(tu[tu.stay_id==2020000000].to_string())
```