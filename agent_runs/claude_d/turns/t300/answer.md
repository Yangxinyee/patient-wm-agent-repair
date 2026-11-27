The z posterior at insulin-only bundles does not reproduce for any unit/rate-unit pair (best error 0.08). This is the same unresolved mismatch as runs 237–238. So my model of insulin bundles is missing something. Either these bundles hold other atoms, or insulin atoms carry extra categorical fields (action, route or status).

This matters for safety. My chosen gate is not constrained orthogonal to the action/route/status/kq/text embeddings. If insulin atoms include any of them, the gate value on insulin could drift toward firing.

Two checks in this run:
- Print every record at the minute of one candidate bundle, to see what it really contains.
- Measure the chosen gate's dot product with each optional categorical embedding, to bound possible drift on insulin atoms.

```python
rw=cand.iloc[0]; g=TRi[TRi.stay_id==rw.stay_id]
allr=pd.read_csv('/data/train.csv'); allr=allr[allr.stay_id==rw.stay_id]
mm=rw.minute; print('bundle minute',mm, 'row',int(rw.row))
print(allr[(np.round(allr.hour*60)==mm)|(np.round(allr.end_hour*60)==mm)].to_string())
gw=gate(-1.0,-4,-3)
for n in ['embed.categorical.action.weight','embed.categorical.route.weight','embed.categorical.status.weight','embed.categorical.discharged_to.weight','embed.kq.weight','embed.text.weight']:
    print(n.split('.')[-2], np.round(Wt(n)@gw,2))
```