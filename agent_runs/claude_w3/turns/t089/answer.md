The row norms show which embedding indices the data actually uses. Rows at about 0.28 are untouched initialisation, rows at 0 are padding, and larger rows were trained.
- **kind:** only 0 (WORLD) and 2 (END) are used.
- **ekind:** the index equals the event-kind id.
- **kq:** always 0.
- **text, action, route, status, discharged_to:** all sit at padding or untrained rows, so they are constant.
- **rate_unit:** two used rows, 2 and 3.
- **unit:** four used rows, 2–5, presumably mg, mg/dL, mmHg and units in some order.

So each atom's embedding is a code-specific constant plus bins and z-scores. The only real unknowns are the unit permutation and what END atoms carry, and both can be checked against the z-posterior.

Next I will record this. Then I will pick ~10 tuning stays from sel100 that have insulin, norepinephrine and arterial readings, so that one `states` pass over all their bundles can validate a local reimplementation step by step. Factual means for these stays are already in fwd_2.

```python
import json
with open('/work/notes.md','a') as fh:
    fh.write("- Embedding tables: used rows: kind{0,2}, ekind=event kind id, kq=0, text/action/route/status/discharged_to unused(const), rate_unit{2,3}, unit{2..5}. Init-norm .28 = unused.\n- Plan: full local reimpl of model, validate vs states (all bundles) + factual means (fwd_2 sel100).\n")
sel = json.load(open('/work/sel100.json'))
sel = sel if isinstance(sel, list) else list(sel.values())[0]
tsel = tu[tu.stay_id.isin(sel)]
cnt_ = tsel.groupby(['stay_id','event']).size().unstack(fill_value=0)
good = cnt_[(cnt_.get('insulin',0)>=4)&(cnt_.get('norepinephrine',0)>=4)&(cnt_.get('map_arterial',0)>=10)].index.tolist()
print(len(sel), len(good), good[:12])
print(os.listdir('/work/fwd_2'), os.listdir('/work/fwd_3'))
```