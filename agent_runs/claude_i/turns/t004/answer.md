Every stay has the same layout: 25.17 hours long, readings on the hour from −1 to 24, and drug records 5 minutes past the hour from 0.08 to 24.08.

- **Norepinephrine:** dose = 4.8 × rate exactly, which corresponds to a fixed 80 kg weight. So dose carries no extra information beyond rate. Rates are capped at 1.0, and 33k of the 242k records sit at the cap.
- **Insulin:** rates come in 0.5 steps from 0.5 to 14.5, and dose = rate exactly.

Next I'll build an hourly panel with one row per stay and hour (−1..24). It will hold the four readings plus the insulin and norepinephrine rates started at h + 5 min. With it I can study three things:
- the dosing policy: what predicts the rate and how deterministic it is;
- missingness patterns of the readings;
- raw dose and outcome associations.

```python
def panel(df):
    df = df.copy()
    df['h'] = np.floor(df.hour+1e-6).astype(int)
    out = {}
    for ev,col in [('glucose_finger','gf'),('glucose_lab','gl'),('map_arterial','ma'),('map_cuff','mc')]:
        d = df[df.event==ev].groupby(['stay_id','h']).value.mean()
        out[col]=d
    for ev,col in [('insulin','ins'),('norepinephrine','ne')]:
        d = df[df.event==ev].groupby(['stay_id','h']).rate.sum()
        out[col]=d
        c = df[df.event==ev].groupby(['stay_id','h']).rate.count()
        out[col+'_n']=c
    P = pd.DataFrame(out)
    stays = df.stay_id.unique()
    idx = pd.MultiIndex.from_product([stays, np.arange(-1,25)], names=['stay_id','h'])
    P = P.reindex(idx)
    return P
P = panel(train); PT = panel(tune)
print(P.shape)
print(P[['ins_n','ne_n']].max())
print("missing frac by col:\n", P.isna().mean())
print("presence by hour:\n", P.notna().groupby(level='h').mean().round(3).T.to_string())
```