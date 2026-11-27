The bundles match the record exactly. Events in the same minute form one bundle. Each infusion produces a start atom at its hour and an END atom one hour later, which lands in the same bundle as the next start. For example, hour 5.0833 has 4 atoms: insulin start, insulin END, norepinephrine start and norepinephrine END.

Before writing any network code I'll check that I can rebuild this tokenisation for all 100 stays. I'll build each stay's bundles from the CSV (measurements, visit events, drug starts, and drug ENDs at end_hour) and compare bundle times and atom counts with the 5,116 rows the model reported. An exact match confirms the bundling. Mismatches would show edge cases, such as END atoms after the stay ends or the admission rows' end_hour.

```python
def build_bundles(df_stay):
    ev = []
    for r in df_stay.itertuples(index=False):
        if r.event in ('insulin','norepinephrine'):
            ev.append((round(r.hour*60), r.event, 'start', r))
            ev.append((round(r.end_hour*60), r.event, 'end', r))
        else:
            ev.append((round(r.hour*60), r.event, 'world', r))
    ev.sort(key=lambda t: t[0])
    out_ = {}
    for t_, e_, k_, r_ in ev:
        out_.setdefault(t_, []).append((e_, k_, r_))
    return out_
mism = []
for s_ in np.unique(sid):
    bb = build_bundles(tu[tu.stay_id==s_])
    ms = sid==s_
    hr_model = np.round(hrs[ms]*60).astype(int)
    keys = sorted(bb.keys())
    if len(keys)!=ms.sum() or not np.all(np.array(keys)==hr_model) or not np.all(np.array([len(bb[k_]) for k_ in keys])==na_[ms]):
        mism.append(s_)
print('mismatching stays', len(mism))
if mism:
    s_ = mism[0]; bb = build_bundles(tu[tu.stay_id==s_]); ms = sid==s_
    keys = sorted(bb.keys())
    print(len(keys), ms.sum()); print(np.array(keys)[-8:], np.round(hrs[ms]*60).astype(int)[-8:])
    print([len(bb[k_]) for k_ in keys][-8:], na_[ms][-8:])
```