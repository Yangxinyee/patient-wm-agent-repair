All 8 `last_numeric` columns and all 15 `type_emb` rows are trained, so none is dead and the layout has to be found by matching. `last_status` row 0 is zero (padding) and row 1 is trained.

The z prior has log-sd about −0.3 to 0, i.e. sd near 1. The head's 4 prior draws therefore add real noise, and a local check must average many draws and compare to the tool loosely.

A plausible 8-input layout mirrors the atom numerics: [has_last, log(1+age)/10, z_value, has_value, z_dose, has_dose, z_rate, has_rate]. For glucose readings only the first four matter. To build the head check from the tool's own states (X_), I first need the exact structure of `bundles_B` entries, the names of the embedding tables, and the row bookkeeping (`row_start`).

```python
sid0 = selB[0]
Bq = bundles_B[sid0]
print(type(Bq), len(Bq), type(Bq[0]))
for k in range(4):
    print(k, str(Bq[k])[:300])
print([n for n in NAMES if n.startswith('embed.')])
print('row_start', row_start[sid0], 'n rows for stay', int((SIDS==sid0).sum()), 'len bundles', len(Bq))
print('vbc keys', list(vbc.keys()), 'numeric keys', list(V['numeric'].keys())[:6], str(V['numeric'][list(V['numeric'].keys())[0]])[:300])
```