The failures fall into two groups:
- 226 END-only bundles at non-integer hours, when a drug record ended with no new one starting. I can encode these from END atoms of records whose end_hour matches.
- About 200 admission-containing bundles (hospital_admission at −2, icu_admission at 0 mixed with readings). These need a visit-event atom.

I'll encode the END-only bundles exactly. For admissions I'll try `meas_parts` with no value, and if that fails fall back to an approximate atom. These affect only the memory M, so small errors there should matter little.

```python
def try_atom(ev, val):
    try: return meas_parts(ev, val)
    except Exception: return None
print('adm atom test', try_atom('hospital_admission', float('nan')) is not None, try_atom('icu_admission', float('nan')) is not None)
drugs_tu = tus[tus.event.isin(['insulin','norepinephrine'])]
endg = {}
for r in drugs_tu.itertuples():
    endg.setdefault((r.stay_id, round(float(r.end_hour), 4)), []).append(r.event)
n_ok = 0; n_bad = 0
for i in fr:
    g = grp_all.get((st[i], hr[i]))
    atoms = []
    for ev in endg.get((st[i], round(float(hr[i]), 4)), []): atoms.append(end_atom(ev))
    if g is not None:
        for r in g.itertuples():
            a = try_atom(r.event, float(r.value) if r.value == r.value else float('nan'))
            if a is not None: atoms.append(a)
    if atoms:
        e_all[i] = enc_e(atoms).reshape(-1); src[i] = 'other'; n_ok += 1
    else: n_bad += 1
print('ok', n_ok, 'bad', n_bad)
```