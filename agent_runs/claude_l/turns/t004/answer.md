The tensor list confirms the path a dose edit takes. Drug atoms are embedded through tables shared between insulin and norepinephrine (`embed.numeric` z_dose/z_rate columns, `embed.bins.dose`/`embed.bins.rate`), go through `encode_action.*` into u_k, and u_k feeds `z_post` (input [ctx, e_k, u_k], 576 wide) and the prelude. The records show readings at whole hours and insulin starting 5 minutes later and running one hour.

Before making any causal claim I need to know how clinicians chose the doses. If doses depend only on recorded readings plus independent noise, the drug effects are identifiable by regression with adjustment for history. If they depend on something unrecorded, the dose is itself evidence about hidden state, which is how a model ends up learning association instead of effect.

I'll start the notebook, then look at stay-level structure: which stays get which drugs, timing regularity, the possible rate values, and whether the stay_id encodes anything.

```python
notes = """# NOTES (world model dose-effect repair)

## Setup facts (run 1-3)
- Model: event-stream, d=192. Dose edit acts ONLY via drug atoms' rate/dose features -> embed (embed.numeric cols z_dose(2), z_rate(4);
  embed.bins.dose / embed.bins.rate tables, shared insulin/NE) -> encode_action (DeepSets) -> u_k -> z_post([ctx,e_k,u_k]) & prelude token.
- Insulin dose/rate: median 1.5, iqr 1.0, bin edges up to 5.0 (saturates). NE rate: median .46 iqr .59, edges to 1.0; NE dose median 2.208 iqr 2.832 edges to 4.8.
- Data: readings at integer hours; infusion records start at hour+0.0833 and last 1h.
"""
open('/work/notes.md','w').write(notes)

# stay-level structure
g = tr.groupby('stay_id')
ev_counts = tr.pivot_table(index='stay_id', columns='event', values='hour', aggfunc='count').fillna(0)
print(ev_counts.describe().T)
print("frac stays with insulin:", (ev_counts['insulin']>0).mean(), " NE:", (ev_counts['norepinephrine']>0).mean())
print(pd.crosstab(ev_counts['insulin']>0, ev_counts['norepinephrine']>0))
# stay length
adm = tr[tr.event=='icu_admission'][['stay_id','end_hour']].set_index('stay_id')
print(adm.end_hour.describe())
# hour fractional parts per event
tr['frac'] = (tr.hour - np.floor(tr.hour)).round(4)
print(tr.groupby('event').frac.value_counts().groupby(level=0).head(5))
# stay id structure
sid = tr.stay_id.unique()
print(sid[:20], sid[-20:])
print(pd.Series(sid//10000000).value_counts().head(20))
```