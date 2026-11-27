The README gives the units: glucose mg/dL, MAP mmHg, insulin dose units, norepinephrine dose mg, and rate units units/hour and mcg/kg/min. So the 6-row unit table is padding, unknown and these 4 units in some order, and the 4-row rate-unit table is padding, unknown and the 2 rate units. cfg0 used frequency order (mmHg 2, mg/dL 3, units 4, mg 5), which is the natural guess. I'll test all orderings cleanly once the bundles are right.

First I'll rebuild the bundles from the record for the 100 fwd_3 stays:
- each measurement is one atom;
- each infusion record gives a START atom at its hour and an END atom at its end_hour;
- the admission rows are visit atoms;
- atoms are grouped by stay and minute.

Then I'll check that my bundles match the tool's rows in count, hour, n_atoms and the insulin and norepinephrine flags. Full agreement would confirm the bundling, including END atoms. Mismatches would show which events I'm missing.

```python
def build_bundles(df):
    rows = []
    for r in df.itertuples(index=False):
        ev = r.event
        if ev in ('insulin','norepinephrine'):
            rows.append((r.stay_id, round(r.hour*60), ev, 0, np.nan, r.rate, r.dose))
            rows.append((r.stay_id, round(r.end_hour*60), ev, 2, np.nan, r.rate, r.dose))
        else:
            rows.append((r.stay_id, round(r.hour*60), ev, 0, r.value, np.nan, np.nan))
    A_ = pd.DataFrame(rows, columns=['stay_id','minute','code','kind','value','rate','dose'])
    A_ = A_.sort_values(['stay_id','minute']).reset_index(drop=True)
    return A_
A3 = build_bundles(tr3)
bk = A3.groupby(['stay_id','minute'])
bsum = bk.agg(n=('code','size'), ins=('code', lambda c: (c=='insulin').any()), ne=('code', lambda c: (c=='norepinephrine').any())).reset_index()
print(len(bsum), 'tool rows', len(sid3))
tool = pd.DataFrame({'stay_id': sid3, 'minute': np.round(hr3*60).astype(int), 'n_tool': z3['n_atoms'], 'ins_t': z3['has_insulin'], 'ne_t': z3['has_norepinephrine']})
mg = tool.merge(bsum, on=['stay_id','minute'], how='outer', indicator=True)
print(mg._merge.value_counts().to_dict())
both_ = mg[mg._merge=='both']
print('n match', (both_.n==both_.n_tool).mean(), 'ins match', (both_.ins==both_.ins_t).mean(), 'ne match', (both_.ne==both_.ne_t).mean())
print(mg[mg._merge!='both'].head(6).to_string())
print(both_[both_.n!=both_.n_tool].head(6).to_string())
```