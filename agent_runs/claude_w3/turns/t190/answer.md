`flow.a` gives timescales from 10 minutes to 2 years with dt in seconds, as expected. Next I'll compute slot-to-action-token attention per head over the 12 tuning stays' drug bundles, and report its mean and spread.

```python
Hs=S4['x'][:,192:960].reshape(-1,4,192); Zs=S4['x'][:,960:]
PU=[];PE=[]
for sid in np.unique(S4['stay_id']):
    rows=np.where(S4['stay_id']==sid)[0]; Bs=stay_bundles(int(sid))
    for k in range(1,len(rows)):
        bb=Bs[k]
        if not bb['code'].isin([8,9]).any(): continue
        e,u,_,_=enc_bundle(bb); dt=(bb['minute'].iloc[0]-Bs[k-1]['minute'].iloc[0])*60
        X=pre_tokens(torch.tensor(Hs[rows[k-1]]).double(),dt,e.double(),u.double(),torch.tensor(Zs[rows[k]]).double())
        P=pre_probs(X); PU.append(P[:,:4,5].numpy()); PE.append(P[:,:4,4].numpy())
PU=np.array(PU); PE=np.array(PE)
print(PU.shape, 'p(u) mean by head', PU.mean((0,2)).round(3), 'sd', PU.std((0,2)).round(3))
print('p(e) mean by head', PE.mean((0,2)).round(3))
```