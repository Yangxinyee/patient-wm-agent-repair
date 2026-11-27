The states request returned 5,345 bundle rows for the 100 training stays, about 53 per stay. Each row holds out, the four h slots and the z posterior mean, plus the stay, hour, drug flags and atom count.

First I'll check that my reconstruction of bundles from the CSV lines up with the model's, before reimplementing anything:
- one bundle per distinct minute;
- infusion END atoms at start + 1 h;
- admission END atoms at the stay end.

If my bundle hours and atom counts match the tool's, I've understood the input format. If counts differ, they'll show which atoms (END, admission ends) I'm missing.

```python
S = np.load('/work/fwd_3/states_original.npz')
print({k: S[k].shape for k in S.files})
st_ids = S['stay_id']; hrs = S['hour']; nat = S['n_atoms']
sid0 = selT[0]
mask = st_ids==sid0
print(np.c_[hrs[mask], nat[mask], S['has_insulin'][mask], S['has_norepinephrine'][mask]][:40])
print(tr[tr.stay_id==sid0][['hour','event','value','rate','end_hour']].head(40).to_string())
```