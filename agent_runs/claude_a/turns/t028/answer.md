Forward pass 3 returned 5,369 bundle states for the 100 stays, about 54 per stay. Each row holds out (192), four h slots (4×192) and the z posterior mean (32), plus stay, hour, drug flags and atom counts.

Next I'll inspect the bundle structure: the hours, atom counts, and whether infusion END atoms form their own bundles. I'll also check that the rows line up with the record events. Then I'll test whether the most recent insulin rate is linearly decodable from the state right after the insulin bundle, which is the state used to predict the next glucose. I'll use ridge regression with cross-validation, and do the same for norepinephrine. High decodability makes a head-level correction feasible.

```python
S = np.load('/work/fwd_3/states_original.npz', allow_pickle=True)
print({k:S[k].shape for k in S.files})
st = pd.DataFrame({'stay_id':S['stay_id'],'hour':S['hour'],'ins':S['has_insulin'],'ne':S['has_norepinephrine'],'n_atoms':S['n_atoms']})
print(st.head(30).to_string())
print(st.groupby(['ins','ne']).size())
print(st.n_atoms.value_counts().sort_index().to_dict())
# compare with record: events per bundle for one stay
s0 = st.stay_id.iloc[0]; rec = tr[tr.stay_id==s0]
print(rec.groupby('hour').event.apply(list).head(20).to_string())
```