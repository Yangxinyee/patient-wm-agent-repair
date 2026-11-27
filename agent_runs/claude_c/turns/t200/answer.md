Removing the action token entirely (pass 15) moves finger-glucose predictions after insulin by less than 1 mg/dL, and partly in the wrong direction. The model barely uses the token for glucose. Two unrelated edits (numeric ×5 and head-0 ×4) both stopped near −2.5. That suggests either a ceiling downstream of the token or a coincidence. Next I will regress the per-stay-hour shift on the extra insulin received since hour 12, for passes 1, 6 and 18. A ceiling would show as flattening at large extra doses.

```python
recs=ins.copy()
def extra(df):
    r=recs[recs.stay_id.isin(df.stay_id.unique())&(recs.hour>=12)]
    out=[]
    for s,h in zip(df.stay_id,df.hour):
        out.append(r[(r.stay_id==s)&(r.hour<h)].rate.sum())
    return np.array(out)
for k,mdl in [(1,'original'),(6,'edited'),(18,'edited')]:
    df=pd.read_csv(f'/work/fwd_{k}/dose_edit_{mdl}.csv'); df=df[(df.edit=='insulin_x2')&df.reached.astype(bool)].copy()
    df['x']=extra(df); df['xb']=pd.cut(df.x,[0,3,6,10,15,25,60])
    print(k, df.groupby('xb',observed=True)['shift'].agg(['size','mean']).round(2).T.values.tolist())
```