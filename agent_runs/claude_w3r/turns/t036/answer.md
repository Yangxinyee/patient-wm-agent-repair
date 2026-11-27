The bundle structure is clear: readings at integer hours, drug records at hour + 0.0833, final END atoms at 25.08 and the stay end at 25.17. The insulin rate can be decoded from the state right after an insulin bundle with R² ≈ 0.65–0.69. That may come mostly from glucose (the band) rather than the dose itself.

The decisive question is whether the state encodes the random deviation r − E[r | band]. That deviation is what the edit must use: it is uncorrelated with glucose history, so using it does not harm the factual fit. Next I will probe the band-relative deviation from the state at the insulin bundle and at the following bundles (t+1 reading, t+1.08, t+2), with a variance-explained check.

```python
gl_last_S = {}
gfS = trS[trS.event=='glucose_finger'][['stay_id','hour','value']]
def last_val(df, sidv, hv):
    d=df[(df.stay_id==sidv)&(df.hour<=hv)]
    return d.value.iloc[-1] if len(d) else np.nan
Bi['g_last']=[last_val(gfS,s,h) for s,h in zip(Bi.stay_id,Bi.hour)]
Bi['g5']=np.floor(Bi.g_last/5)*5; Bi['c']=Bi.g5.map(mp)  # mp: E[r|g5] from full training panel
Bi['dev']=Bi.rate-Bi.c
print('dev sd', Bi.dev.std(), 'rate sd', Bi.rate.std(), 'corr(rate,c)', Bi[['rate','c']].corr().iloc[0,1])
B['ord']=B.groupby('stay_id').cumcount()
for lag in [0,1,2,3,4,6]:
    rows_=[]
    for _,r_ in Bi.iterrows():
        bb=B[(B.stay_id==r_.stay_id)]; j=np.where(bb.hour.values==r_.hour)[0][0]+lag
        rows_.append(bb.row.values[j] if j<len(bb) else -1)
    rows_=np.array(rows_); ok=rows_>=0
    for nm,sl in [('all',slice(0,992))]:
        r2d=probe(Xs[rows_[ok]][:,sl], Bi.dev.values[ok], Bi.stay_id.values[ok])
        print('bundle lag',lag,'R2 dev', round(r2d,3), ' n', ok.sum())
```