The initial state is not the cause: every variant of h0, out0 and gap input still leaves about a 0.88 error. The problem must be in how the bundle is encoded, or in an unstated convention.

There is a better test bed than the first bundle. For any later bundle, the context comes from the previous bundle's out and h, which the states file gives exactly. So z_k = z_post([ctx_k, e_k, u_k]) can be checked on thousands of bundles of every type:
- measurements only, which tests `encode`;
- drugs only, which tests `encode_action`;
- mixed bundles.

I'll write a parser that turns tuning records into bundles of atoms: code, kind, ekind, unit, bins and z-scores, with END atoms placed at end_hour. Then I'll check the one-step z prediction on measurement-only bundles, varying activation and gap scaling. A near-zero error on these would confirm the measurement encoder and ctx formula, and I'd then move on to drug atoms.

```python
NUM = {(r['code'], r['attribute']): r for r in V['numeric']}
CODE_ID = {c['name']: c['id'] for c in V['codes']}
def zscore(code, attr, x):
    r = NUM[(code, attr)]; return float(np.clip((x - r['median'])/r['iqr'], -10, 10))
def fbin(code, attr, x, side='left'):
    return 1 + int(np.searchsorted(np.array(NUM[(code, attr)]['fine_bin_edges']), x, side=side))
def build_bundles(df_stay, unit_map, rate_unit_map, side='left', end_attrs=True):
    """returns list of (time_hours, [atom dicts])"""
    ev = []
    for r in df_stay.itertuples():
        e = r.event
        if e in ('glucose_finger','glucose_lab','map_arterial','map_cuff'):
            ev.append((r.hour, dict(code=CODE_ID[e], kind=0, ekind=1, unit=unit_map[V_unit[e]], rate_unit=0, vb=fbin(e,'value',r.value,side),
                                    db=0, rb=0, num=(zscore(e,'value',r.value),1,0,0,0,0), act=False)))
        elif e in ('insulin','norepinephrine'):
            a = dict(code=CODE_ID[e], ekind=2, unit=unit_map[r.dose_unit], rate_unit=rate_unit_map[r.rate_unit], db=fbin(e,'dose',r.dose,side), rb=fbin(e,'rate',r.rate,side),
                     vb=0, num=(0,0,zscore(e,'dose',r.dose),1,zscore(e,'rate',r.rate),1), act=True)
            ev.append((r.hour, dict(a, kind=0)))
            endA = dict(a, kind=2) if end_attrs else dict(code=CODE_ID[e], ekind=2, unit=0, rate_unit=0, db=0, rb=0, vb=0, num=(0,0,0,0,0,0), act=True, kind=2)
            ev.append((r.end_hour, endA))
        else:
            ek = 3 if e=='hospital_admission' else 4
            a = dict(code=CODE_ID[e], ekind=ek, unit=0, rate_unit=0, vb=0, db=0, rb=0, num=(0,0,0,0,0,0), act=False)
            ev.append((r.hour, dict(a, kind=0))); ev.append((r.end_hour, dict(a, kind=2)))
    ev.sort(key=lambda t: t[0])
    out = []; 
    for t, a in ev:
        key = round(t*60)  # minute
        if out and out[-1][0]==key: out[-1][2].append(a)
        else: out.append((key, t, [a]))
    return [(t, atoms) for key, t, atoms in out]
V_unit = {'glucose_finger':'mg/dL','glucose_lab':'mg/dL','map_arterial':'mmHg','map_cuff':'mmHg'}
um = {'mg/dL':2, 'mmHg':5, 'units':3, 'mg':4}; rum = {'units/hour':2, 'mcg/kg/min':3}
bb = build_bundles(tu[tu.stay_id==2020000000].sort_values('hour'), um, rum)
print(len(bb), [ (round(t,4), len(a)) for t,a in bb[:12]])
print(sx[sx.stay_id==2020000000].n_atoms.values[:12])
```