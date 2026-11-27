Every stay is off by exactly one bundle: the model has an extra 2-atom bundle at the stay's end (25.1667 h). That is the admissions' end_hour. The hospital_admission and icu_admission records also emit END atoms when the stay ends, just like infusions.

I'll add END atoms for the two visit events and redo the comparison. All 100 stays matching on bundle times and atom counts would confirm the tokenisation. Any remaining mismatch would point to another edge case, such as END atoms of infusions that run past the stay's end.

```python
def build_bundles(df_stay):
    ev = []
    for r in df_stay.itertuples(index=False):
        if r.event in ('insulin','norepinephrine','hospital_admission','icu_admission'):
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
# what's at the end of an infusion that runs past stay end?
print(tu.groupby('stay_id').end_hour.max().head())
```