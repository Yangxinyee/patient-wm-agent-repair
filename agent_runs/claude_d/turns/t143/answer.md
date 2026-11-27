Both drugs start at hh:05 and all readings fall on the hour. So each reading at hour t+1 is predicted from the state after the t:05 drug bundle.

Before fitting decoders I need to settle how the correction is centred, and the two drugs differ. Insulin depends only on current glucose plus an independent multiplier, so centring on the protocol's expected dose given the readings works. NE is titrated from the previous rate. If I centre NE on its expectation given the previous dose, a sustained doubling would cancel out and give almost no effect. NE should therefore be centred on what the readings alone predict.

First I'll build an hourly panel from the training records: glucose, MAP, the insulin and NE rates at t:05, and the insulin base level. From it I'll get the exact E[insulin rate | base].

```python
def hourly_panel(df):
    d = df.copy(); d['t'] = np.floor(d['hour']+1e-6).astype(int)
    rd = d[d.event.isin(['glucose_finger','glucose_lab','map_arterial','map_cuff'])].pivot_table(index=['stay_id','t'],columns='event',values='value',aggfunc='first')
    dr = d[d.event.isin(['insulin','norepinephrine'])].pivot_table(index=['stay_id','t'],columns='event',values='rate',aggfunc='first')
    P_ = rd.join(dr,how='outer').reset_index()
    # fill full hour grid per stay
    rng_ = P_.groupby('stay_id')['t'].agg(['min','max'])
    grid = pd.DataFrame([(s,t) for s,(a,b) in rng_.iterrows() for t in range(a,b+1)],columns=['stay_id','t'])
    P_ = grid.merge(P_,on=['stay_id','t'],how='left')
    for c in ['glucose_finger','glucose_lab','map_arterial','map_cuff','insulin','norepinephrine']:
        if c not in P_: P_[c]=np.nan
    P_['ins']=P_['insulin'].fillna(0.0); P_['ne']=P_['norepinephrine'].fillna(0.0)
    P_['Gdec']=P_['glucose_finger'].fillna(P_['glucose_lab'])
    return P_
PT_tr = hourly_panel(TRAIN)
G_=PT_tr['Gdec']; PT_tr['base']=np.select([G_>=300,G_>=260,G_>=220,G_>=180,G_>=150],[6,4,3,2,1],0)
print(len(PT_tr), PT_tr[['ins','ne']].gt(0).mean().round(3).to_dict())
tb = PT_tr[PT_tr.Gdec.notna()].groupby('base')['ins'].agg(['mean','std','size',lambda x:(x>0).mean()]); tb.columns=['mean','sd','n','frac_pos']; print(tb.round(3).to_string())
print('ins>0 with Gdec missing:', ((PT_tr.ins>0)&PT_tr.Gdec.isna()).sum())
```