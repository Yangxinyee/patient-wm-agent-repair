The states request returned 5,294 bundle rows for the 100 stays, about 53 per stay. That fits one reading bundle and one drug bundle per hour, plus admission events.

Next I'll load the file and:
1. match each row to the record by stay and hour, to see the bundle structure (the atom counts show whether drug bundles include END atoms of the previous infusion);
2. check how state norms behave;
3. regress recent doses on the state at the bundle just before each glucose reading.

For the regression: insulin rate in the last drug bundle, the dose 2 hours back, the rolling dose history and the dose residual (rate minus policy mean). For norepinephrine: the rate in the last drug bundle and its residual. Cross-validated R² will show what the state carries.

```python
Z = np.load('/work/fwd_2/states_original.npz')
print(Z.files)
X = Z['x']; st_sid = Z['stay_id']; st_hr = Z['hour']; st_ins = Z['has_insulin']; st_ne = Z['has_norepinephrine']; st_na = Z['n_atoms']
print(X.shape, st_hr[:60])
print(st_na[:60]); print(st_ins[:60].astype(int)); print(st_ne[:60].astype(int))
print("out norm", np.linalg.norm(X[:, :192],axis=1).mean(), " slot norms", [np.linalg.norm(X[:,192+192*s:384+192*s],axis=1).mean() for s in range(4)], " z norm", np.linalg.norm(X[:,960:],axis=1).mean())
print(tr[tr.stay_id==st_sid[0]][['hour','event','value','rate','end_hour']].head(25).to_string())
```