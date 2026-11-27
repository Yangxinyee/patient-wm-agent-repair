I have 5,156 bundle states for the 100 stays, about 52 per stay. Each row holds out (192), the four h slots (768) and the z posterior mean (32), plus the stay, hour, drug flags and atom count.

Before writing any model code I need the exact bundle and atom structure. I'll check this against the raw records for one stay:
- which events share a bundle;
- whether the visit events count as atoms, and whether END atoms appear at hour + 1h05 when an infusion ends;
- how n_atoms compares with what the raw rows predict.

That fixes how a stay is turned into the model's input.

```python
st = np.load('/work/fwd_3/states_original.npz')
print({k: st[k].shape for k in st.files})
X = st['x']; sid_ = st['stay_id']; hr_ = st['hour']; nat = st['n_atoms']; hi_ = st['has_insulin']; hn_ = st['has_norepinephrine']
s0 = selT[6]  # 1010000480
m = sid_==s0
print(pd.DataFrame({'hour':hr_[m], 'n_atoms':nat[m], 'ins':hi_[m], 'ne':hn_[m]}).head(40).to_string())
print(train[train.stay_id==s0].head(40).to_string())
```