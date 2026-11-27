Row 0 of every categorical and bin table is zero, i.e. padding. Rows that were never trained keep their initial norm of about 0.27. From this pattern:

- **Used rows:** kind 0 (WORLD) and kind 2 (END); event kinds 1–4; kq row 0 only.
- **Unit tables:** unit rows 2–5 are the four units; rate_unit rows 2–3 are the two drug rate units.
- **Always index 0 here:** text, action, discharged_to, route and status.
- **last_status:** row 1 is heavily trained (norm 12.3). My validated head used row 0 for measurements, so that's right.

The open questions are the unit/rate_unit row order and whether END atoms carry rate and dose.

Next, I will rebuild every bundle's atom list for the 100 stays from the records and check the atom counts against the tool's `n_atoms`. That confirms the bundle boundaries and how END atoms are generated, including any at stay end. Exact count agreement confirms the bundling; mismatches will show which events I'm missing.

```python
recs = train[train.stay_id.isin(set(S3))].copy()
atoms_by_bundle = defaultdict(list)   # key (stay, minute) -> list of dict
def mkey(s,h): return (s, int(round(h*60)))
for r in recs.itertuples(index=False):
    ev = r.event
    if ev in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
        atoms_by_bundle[mkey(r.stay_id,r.hour)].append(dict(code=ev, kind=0, value=r.value))
    elif ev in ('insulin','norepinephrine'):
        atoms_by_bundle[mkey(r.stay_id,r.hour)].append(dict(code=ev, kind=0, rate=r.rate, dose=r.dose))
        atoms_by_bundle[mkey(r.stay_id,r.end_hour)].append(dict(code=ev, kind=2, rate=r.rate, dose=r.dose))
    else:
        atoms_by_bundle[mkey(r.stay_id,r.hour)].append(dict(code=ev, kind=0))
        if not np.isnan(r.end_hour): atoms_by_bundle[mkey(r.stay_id,r.end_hour)].append(dict(code=ev, kind=2))
mine_n = np.array([len(atoms_by_bundle.get(mkey(s,h),[])) for s,h in zip(S3,H3)])
print("match", (mine_n==z3['n_atoms']).mean())
bad = np.where(mine_n!=z3['n_atoms'])[0][:10]
for i in bad: print(S3[i], H3[i], mine_n[i], z3['n_atoms'][i], [ (a['code'],a['kind']) for a in atoms_by_bundle.get(mkey(S3[i],H3[i]),[])])
print("bundles per stay", len(S3)/100, "my bundles not in states:", len(set(k for k in atoms_by_bundle) - set(mkey(s,h) for s,h in zip(S3,H3))))
```