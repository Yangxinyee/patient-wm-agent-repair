Each stay runs from hour -2 (hospital admission) to 25.17. Readings fall on whole hours from -1 to 24, and infusions start at h + 0.083 for h from 0 to 24. Insulin dose equals its rate. Norepinephrine dose is 4.8 x rate, implying an 80 kg weight. About 61% of stays have insulin and 44% have norepinephrine. Insulin rates lie on a 0.5 grid; norepinephrine rates on a 0.01 grid capped at 1.0.

So everything can be put in an hourly panel: per stay and hour, the four readings plus the insulin and norepinephrine rates that start just after that hour. Next I will build the panel and look at how doses relate to the latest readings: when insulin is started, and whether the rate is a function of the last glucose_finger or glucose_lab. If the rate is a deterministic function of observed readings, adjusting for those readings will identify the effect. If it has extra variation, I need to find out whether that variation is random or confounded.

```python
def panel(df):
    df = df.copy()
    df['h'] = np.floor(df.hour).astype(int)
    keep = df[df.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])]
    P = keep.pivot_table(index=['stay_id','h'], columns='event', values='value', aggfunc='first')
    drugs = df[df.event.isin(['insulin','norepinephrine'])]
    D = drugs.pivot_table(index=['stay_id','h'], columns='event', values='rate', aggfunc='first')
    stays = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([stays, range(-1,25)], names=['stay_id','h'])
    out = pd.DataFrame(index=idx).join(P).join(D)
    out.columns = [ {'glucose_finger':'GF','glucose_lab':'GL','map_arterial':'MA','map_cuff':'MC','insulin':'I','norepinephrine':'N'}[c] for c in out.columns]
    return out.reset_index()
P = panel(tr); PU = panel(tu)
print(P.head(30).to_string())
print(P.notna().groupby(P.h).mean().round(2).to_string())
```