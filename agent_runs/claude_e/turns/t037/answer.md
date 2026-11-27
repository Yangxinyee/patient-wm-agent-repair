The embedding tables show which inputs are actually used:
- Row 0 of most categorical tables is exactly zero (padding).
- Rows with norm about 0.28 were never trained.
- Used units: rows 2–5 of `unit` (four units) and rows 2–3 of `rate_unit`.
- Atom kinds 0 and 2 are used (WORLD, END); `kq` row 0 is always added; text, status, route, action and discharged_to are absent.

One finding matters directly for the x2 edit: the insulin dose-bin rows 13–16 (doses of 3.5 u/h and up) are barely trained, with norms 0.27–0.39. Doubling sends many records into those bins.

The time-flow timescales span 600 s to 2 years, so dt is in seconds.

Plan for a local re-implementation, checked in stages against the tool's states:
1. Compute the z posterior mean from the tool's previous out/h plus my own encoding of bundle k. This isolates the encoders, the gap MLP and `z_post`.
2. Then the full core step, which needs the memory of earlier bundle vectors.

This run loads the weights and builds bundles and atoms from the records, with explicit switches for the unknown conventions (unit indices, END-atom attributes).

```python
import torch, torch.nn.functional as Fnn
torch.set_grad_enabled(False)
W = {n: load_original(n).float() for n in NAMES}
code_id = {c['name']: c['id'] for c in V['codes']}
num = {(e['code'], e['attribute']): e for e in V['numeric']}
def fine_bin(code, attr, x):
    e = num.get((code, attr)); 
    if e is None or x is None or np.isnan(x): return 0
    return 1 + int(np.sum(np.array(e['fine_bin_edges']) < x))
def zsc(code, attr, x):
    e = num.get((code, attr)); 
    if e is None or x is None or np.isnan(x): return 0.0, 0.0
    return float(np.clip((x - e['median'])/e['iqr'], -10, 10)), 1.0
EK = {'glucose_finger':1,'glucose_lab':1,'map_arterial':1,'map_cuff':1,'insulin':2,'norepinephrine':2,'hospital_admission':3,'icu_admission':4}
def stay_bundles(df):
    """list of (time_h, [atoms]); atom = dict(code, kind, ev, value, rate, dose)"""
    ev = {}
    for r in df.itertuples():
        a = dict(code=r.event, kind=0, value=r.value, rate=r.rate, dose=r.dose)
        ev.setdefault(round(r.hour,4), []).append(a)
        if r.event in ('insulin','norepinephrine','hospital_admission','icu_admission') and not np.isnan(r.end_hour):
            ev.setdefault(round(r.end_hour,4), []).append(dict(code=r.event, kind=2, value=r.value, rate=r.rate, dose=r.dose))
    return [(t, ev[t]) for t in sorted(ev)]
tu_g = {sid: g for sid, g in tu.groupby('stay_id')}
B0 = stay_bundles(tu_g[2020000000])
print(len(B0), [ (t, len(a)) for t,a in B0][:12])
print('tool n_atoms:', S2['n_atoms'][S2['stay_id']==2020000000][:12])
```