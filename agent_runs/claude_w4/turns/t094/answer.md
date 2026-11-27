My `reached` set contains all of the tool's reached stay-hours but has extra ones (1033 vs 919 for insulin, 1107 vs 730 for norepinephrine). The tool probably counts only hours that actually have an outcome reading. For example, stays without an arterial line have no map_arterial readings.

The next run does three things:
- Restrict my truth to the tool's reached rows, so model and truth are compared on the same stay-hours.
- Check that guess about which rows the tool keeps.
- Compute the other scored interventions (x0, x1.5, x3, start hours 6 and 16) to see the required linearity and time profile.

```python
def truth_on(df, drug, irf, mult, keys, start=12):
    sids_ = sorted(set(k[0] for k in keys)); R = hourly_rates(df, drug, sids_); pos = {s:i for i,s in enumerate(sids_)}
    out_ = []
    for sid_, t in keys:
        i = pos[sid_]; dR = np.where(np.arange(26) >= start, (mult-1)*R[i], 0.0)
        out_.append(sum(irf[l-1]*dR[t-l] for l in range(1,13) if t-l >= 0))
    return np.array(out_)
T_ = {}
for drug, irf, ed2, outc in [('insulin', hF, 'insulin_x2', 'glucose_finger'), ('norepinephrine', hA, 'ne_x2', 'map_arterial')]:
    keys = [tuple(x) for x in de1[(de1.edit==ed2) & de1.reached][['stay_id','hour']].values.tolist()]
    # check: does each reached row have an outcome reading in that hour?
    obs = tu[(tu.event==outc)]; obsk = set(zip(obs.stay_id, np.floor(obs.hour+1e-6).astype(int)))
    T_[drug] = {m: truth_on(tu, drug, irf, m, keys).mean() for m in [0, 0.5, 1.5, 2, 3]}
    print(drug, {m: round(v,2) for m,v in T_[drug].items()}, 'reached rows with reading: %d/%d' % (sum(k in obsk for k in keys), len(keys)))
    nr = [tuple(x) for x in de1[(de1.edit==ed2) & ~de1.reached][['stay_id','hour']].values.tolist()]
    print('   non-reached rows', len(nr), 'with reading', sum(k in obsk for k in nr))
```