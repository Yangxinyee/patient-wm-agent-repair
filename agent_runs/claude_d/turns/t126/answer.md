The bundle structure is now clear. Each infusion row starts an hour-long record with a matching END atom at `end_hour`, and admission rows end at the stay end. That explains the earlier finding of 2-atom bundles at hour fraction 0.17. To reproduce head predictions offline, I'll rebuild every stay's bundle sequence (WORLD plus END atoms, grouped by minute). Then I'll check it against the 5,192 state rows from fwd_3 (stay, minute, atom count). If counts and times match row for row, I can line states up with readings exactly.

```python
selS = json.load(open('/work/selS.json')); selS=[int(s) for s in selS]
TRAIN = pd.read_csv('/data/train.csv')
def bundles_of(df):
    w = df.copy(); w['akind']=0; w['minute']=np.round(w['hour']*60).astype(int)
    e = df[df['end_hour'].notna()].copy(); e['akind']=2; e['minute']=np.round(e['end_hour']*60).astype(int)
    for c in ['value','rate','dose']: e[c]=np.nan
    a = pd.concat([w,e]).sort_values(['stay_id','minute','akind']).reset_index(drop=True)
    return a
ATS = bundles_of(TRAIN[TRAIN.stay_id.isin(selS)])
bc = ATS.groupby(['stay_id','minute']).size().reset_index(name='n')
st = pd.DataFrame({'stay_id':S3['stay_id'].astype(int),'minute':np.round(S3['hour'].astype(np.float64)*60).astype(int),'n_atoms':S3['n_atoms']})
print(len(bc), len(st))
m = st.merge(bc, on=['stay_id','minute'], how='outer', indicator=True)
print(m['_merge'].value_counts().to_dict(), (m['n']!=m['n_atoms']).sum())
print(m[m['n']!=m['n_atoms']].head(8).to_string())
```