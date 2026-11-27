The structure is simple.
- stay_ids are sequential and carry no site information. Tuning stays have a different prefix (202…).
- Every stay runs from hour −2 (hospital admission) through hour 24.
- Readings are taken each hour with partial coverage: glucose_finger about 55–60% of stay-hours, glucose_lab 20%, map_arterial 46%, map_cuff 61%.
- Norepinephrine runs in about 9.9k stay-hours per hour, which is flat. Insulin grows from 10.3k to 13.7k over the stay.

Next I will build an hourly panel: per stay and hour, each reading and each drug rate (0 if none). With it I can characterise the dosing policy and the readings' dynamics. The questions are:
- Is map_arterial available for a fixed subset of stays?
- Does norepinephrine run continuously once started?
- How does insulin start and stop relative to glucose?
- How does the rate relate to the current reading and the previous rate?

I will also save the panel to /work for reuse.

```python
def panel(df):
    df = df.copy()
    df['ih'] = np.floor(df.hour+1e-6).astype(int)
    out = {}
    for ev in ['glucose_finger','glucose_lab','map_arterial','map_cuff']:
        x = df[df.event==ev].groupby(['stay_id','ih']).value.mean()
        out[ev] = x
    for ev in ['insulin','norepinephrine']:
        x = df[df.event==ev].groupby(['stay_id','ih']).rate.sum()
        out[ev] = x
        # also exact hour offset
        out[ev+'_t'] = df[df.event==ev].groupby(['stay_id','ih']).hour.min() - df[df.event==ev].groupby(['stay_id','ih']).ih.min()
    P = pd.DataFrame(out)
    idx = pd.MultiIndex.from_product([df.stay_id.unique(), range(-1,25)], names=['stay_id','ih'])
    P = P.reindex(idx)
    return P
P = panel(tr); PT = panel(tu)
P.to_pickle('/work/panel_train.pkl'); PT.to_pickle('/work/panel_tune.pkl')
print(P.head(30).to_string())
print(P[['insulin_t','norepinephrine_t']].describe().to_string())
# per-stay: fraction of hours with map_arterial, norepi
fs = P.groupby(level=0).agg(lambda x: x.notna().mean())
for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
    print(c, np.histogram(fs[c], bins=[0,0.01,0.2,0.4,0.6,0.8,0.99,1.01])[0])
```