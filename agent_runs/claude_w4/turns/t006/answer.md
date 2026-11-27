## Step 5: build an hourly panel

**What the last result showed.**
- Readings are intermittent: each type is missing in many hours.
- In the second stay, insulin rates jump around a lot (4.5 → 2.5 → 1.0 → 2.5 → 3.5 …) while glucose stays near 200. Insulin dosing therefore has large hour-to-hour variation that glucose does not explain.
- Norepinephrine is titrated smoothly upward as MAP falls, reaching the 1.0 cap.
- Once started, both drugs appear to run every hour.

**What this suggests.**
- The insulin noise could be a usable exogenous component, if it is not driven by hidden patient state.
- For norepinephrine, the smooth titration and the cap need a different look.

**What I will do next.** Build a stay × hour panel (hours -2 to 25) holding:
- every reading;
- the insulin and norepinephrine rates (0 when no record);
- start and stop patterns (does a drug, once started, continue to the end?).

This panel is the basis for estimating both the policy and the effects.

**What would count as evidence.** If drugs never stop once started and start in response to readings, the start decision is the main confounded event. Hour-to-hour rate changes within a running infusion could then serve as the identifying variation.

```python
def panel(df):
    df = df.copy()
    df['h'] = np.floor(df.hour+1e-9).astype(int)
    meas = df[df.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])]
    P = meas.pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
    D = df[df.event.isin(['insulin','norepinephrine'])].pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='first')
    D.columns=['ins','ne']
    sids = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([np.sort(sids), np.arange(-2,26)], names=['stay_id','h'])
    P = P.reindex(idx).join(D.reindex(idx))
    P[['ins','ne']] = P[['ins','ne']].fillna(0.0)
    return P
t0=time.time()
P = panel(tr); PT = panel(tu)
print(time.time()-t0, P.shape)
print("drug hour range ins:", P[P.ins>0].reset_index().h.agg(['min','max']).to_dict(), " ne:", P[P.ne>0].reset_index().h.agg(['min','max']).to_dict())
# start/stop patterns
def runs(x):
    on = (x>0).astype(int)
    return (np.diff(on)!=0).sum()
g = P.reset_index()
for d in ['ins','ne']:
    sw = g[g.h.between(0,24)].groupby('stay_id')[d].apply(lambda s: runs(s.values))
    st = g[(g[d]>0)].groupby('stay_id').h.min()
    print(d, "switch count dist:", sw.value_counts().sort_index().head(10).to_dict())
    print(d, "start hour dist:", st.value_counts().sort_index().to_dict())
# measurement frequency per hour
print(g[g.h.between(0,24)][['glucose_finger','glucose_lab','map_arterial','map_cuff']].notna().mean())
```