Run 73 shows how the atom table is built:
- Every record becomes a WORLD atom (kind 0) at its hour.
- Every infusion record also gets an END atom (kind 2) one hour later, carrying the same rate and dose.
- Each admission row gets an END atom at stay end (25.1667 h).
- Atoms recorded in the same minute form a bundle.
- dt uses the exact hours: 0.0833 h → 1 h gives 3300.12 s.

`R` (forward 2) holds per-reading predicted means and NLLs, together with the register fields (`lastv`, `lasth`, `dt`, `row_prev`).

Next I'll write `make_atoms(df, stays)` and a bundle table, so I can encode any stays. I'll check that they reproduce `AT` and `SB` for the sel2 stays, including dt, row order and encodings. Exact agreement of `e`/`u` and `dts` would confirm the construction. Any mismatch would show which rule I have wrong.

```python
def make_atoms(df, stays):
    d = df[df.stay_id.isin(set(stays))]
    w = d[['stay_id','hour','event','value','rate','dose']].copy(); w['kind'] = 0
    en = d[d.event.isin(['insulin','norepinephrine','hospital_admission','icu_admission'])][['stay_id','end_hour','event','rate','dose']].copy()
    en = en.rename(columns={'end_hour':'hour'}); en['value'] = np.nan; en['kind'] = 2
    at = pd.concat([w, en[['stay_id','hour','event','value','rate','dose','kind']]], ignore_index=True)
    at['minute'] = np.round(at['hour'].values*60).astype(int)
    order = {s:i for i,s in enumerate(stays)}
    at['so'] = at['stay_id'].map(order)
    at = at.sort_values(['so','minute','kind'], kind='stable').reset_index(drop=True)
    bk = at[['so','minute']].drop_duplicates().reset_index(drop=True); bk['row'] = np.arange(len(bk))
    at = at.merge(bk, on=['so','minute'], how='left')
    bh = at.groupby('row').agg(stay_id=('stay_id','first'), minute=('minute','first'), hour=('hour','min'), n_atoms=('event','size')).reset_index()
    bh['prev'] = bh['row'] - 1; first = bh['stay_id'].ne(bh['stay_id'].shift()); bh.loc[first,'prev'] = -1
    bh['dts'] = np.where(first, np.nan, (bh['hour'] - bh['hour'].shift())*3600.0)
    return at.drop(columns=['so']), bh
stays_sel2 = list(pd.unique(SB['stay_id']))
at2, bh2 = make_atoms(tr, stays_sel2)
print(len(at2), len(AT), len(bh2), len(SB))
e2_, u2_ = encode_bundles(at2, list(range(len(bh2))))
print("e max err", float((e2_-eALL).abs().max()), "u max err", float((u2_-uALL).abs().max()))
print("dts max err", np.nanmax(np.abs(bh2['dts'].values - SB['dts'].values)), "minute eq", bool((bh2['minute'].values==SB['minute'].values).all()))
```