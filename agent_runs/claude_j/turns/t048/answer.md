The states file holds 5,116 bundles for the 100 stays, about 51 per stay. Each row has out, the 4 slots of h and the z posterior mean, plus hour, insulin/norepinephrine flags and the atom count.

Before any replica work I need the exact bundling. I'll print the first stay's bundle list next to its raw record events. That shows how events group into bundles, and whether an infusion's END atom shares a bundle with the next infusion's start, giving n_atoms = 2. It also checks that the record's minute grouping matches the bundle count.

```python
S = np.load('/work/fwd_3/states_original.npz')
print({k_: S[k_].shape for k_ in S.files})
X = S['x'] if 'x' in S.files else None
sid = S['stay_id']; hrs = S['hour']; hi_ = S['has_insulin']; hn_ = S['has_norepinephrine']; na_ = S['n_atoms']
s0 = sid[0]
m0 = sid==s0
print(pd.DataFrame({'hour':hrs[m0], 'ins':hi_[m0], 'ne':hn_[m0], 'n':na_[m0]}).head(40).to_string())
r0 = tu[tu.stay_id==s0].sort_values('hour')
print(r0.head(40).to_string())
```