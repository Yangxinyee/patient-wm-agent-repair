Several conventions are now clear:
- **Time units:** core.flow.a gives timescales from 575 s to 6.7×10⁷ s, so dt is in seconds.
- **Unused tables:** many embedding rows have norm 0.27-0.28, i.e. still at initialisation. Text, route, action and status are effectively unused, as are kind REVEAL, unit row 1 and rate_unit row 1.
- **Used rows:** kind WORLD and END, all four event kinds, unit rows 2-5, rate_unit rows 2-3, and a constant kq row 0.
- **last_numeric:** all 8 inputs carry trained weights, so the register has more fields than the document lists.

To design an edit precisely I want ground-truth internal states to check a local reimplementation against, piece by piece. The first piece is the action encoder plus z_post, checked against the z posterior mean of bundles that hold only actions. I'll pick 100 training stays with both drugs after hour 12, so the same states can later be used to regress how doses move the state. This run only selects and saves the IDs.

```python
tti = g[g.h>=12].groupby('stay_id').agg(ins=('ins', lambda s: s.notna().sum()), nor=('nor', lambda s: s.notna().sum()))
tr_both = tti[(tti.ins>=6)&(tti.nor>=6)].index.tolist()
rng = np.random.default_rng(0)
selT = [int(s) for s in rng.choice(tr_both, 100, replace=False)]
json.dump(selT, open('/work/selT.json','w'))
print(len(tr_both)); print(json.dumps(selT))
```