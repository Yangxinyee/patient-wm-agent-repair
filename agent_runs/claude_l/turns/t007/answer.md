The schedule tells me how the treatments are structured.

- **Insulin:** half of the 18,335 insulin stays start at hour 0. Starts trickle in at every later hour, and only about 50% of insulin stays run without a gap, so insulin is switched on and off as well as titrated.
- **Norepinephrine:** about 73% of NE stays start at hour 0 and 88% run continuously.
- **Readings:** map_arterial appears in a fixed ~13,700 stays every hour, which looks like an arterial-line subgroup. glucose_finger readings become more frequent over the stay, and glucose_lab is about 6,000 per hour.

To model dosing and dynamics I'll build an hourly panel per stay: each reading type at hour t, plus the insulin and NE rates started at t+5 min (0 if none). From it I'll check:
- (a) whether map_arterial stays are the NE stays;
- (b) how often each reading type is present when a dose is chosen;
- (c) first-pass dosing-policy fits: how the insulin rate relates to the current glucose and to the previous rate.

If the insulin rate is almost deterministic given recorded readings, there is little room for hidden-state confounding in teacher-forced prediction. Large unexplained variation would be the source of either a natural experiment or confounding.

```python
def panel(df):
    df = df.copy()
    df['ih'] = np.floor(df.hour).astype(int)
    meas = df[df.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])]
    P = meas.pivot_table(index=['stay_id','ih'], columns='event', values='value', aggfunc='first')
    dr = df[df.event.isin(['insulin','norepinephrine'])].pivot_table(index=['stay_id','ih'], columns='event', values='rate', aggfunc='first')
    stays = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([stays, range(-1,25)], names=['stay_id','ih'])
    P = P.reindex(idx).join(dr.reindex(idx))
    P[['insulin','norepinephrine']] = P[['insulin','norepinephrine']].fillna(0.0)
    return P
P = panel(tr); PT = panel(tu)
print(P.shape, PT.shape)
print(P.notna().groupby(level=1).mean().round(3).T.iloc[:, :8])
# arterial line stays vs NE stays
st = P.groupby(level=0).agg(has_art=('map_arterial', lambda x: x.notna().any()), has_ne=('norepinephrine', lambda x: (x>0).any()), has_ins=('insulin', lambda x:(x>0).any()))
print(pd.crosstab(st.has_art, st.has_ne)); print(pd.crosstab(st.has_ins, st.has_art))
```