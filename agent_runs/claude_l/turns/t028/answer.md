The states file has 4,932 bundle rows for 100 stays, about 49 per stay, with stay, hour, drug flags and atom counts per bundle.

Next I'll rebuild the bundles from the tuning records and match them to these rows. The plan is to group events by minute and to add END atoms at each infusion's end_hour. If END atoms land in the same minute as the next infusion start, they share its bundle, and an END with no new infusion makes its own bundle. Matching bundle counts, hours and n_atoms per stay will confirm the featurization assumptions. Mismatches will show where END atoms or admission rows are handled differently.

```python
S = np.load('/work/fwd_2/states_original.npz')
print({k: S[k].shape for k in S.files})
sx = pd.DataFrame({'stay_id': S['stay_id'], 'hour': S['hour'], 'ins': S['has_insulin'], 'ne': S['has_norepinephrine'], 'n_atoms': S['n_atoms']})
print(sx.head(40).to_string())
print(sx.groupby('stay_id').size().describe())
```