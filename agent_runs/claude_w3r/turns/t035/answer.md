I received the states: 5,301 bundles for the 100 stays, each row holding out (192), the four h slots (4 × 192) and the z posterior mean (32). Next I will load them, align each bundle with the record (hour, atoms), and run first linear probes. Can the insulin rate be decoded from the state right after an insulin bundle, and does it persist in the state at later bundles? High R² that holds over hours would mean dose information is available for a head-side edit.

```python
Z = np.load('/work/fwd_2/states_original.npz')
print({k:Z[k].shape for k in Z.files})
Xs=Z['x']; sid=Z['stay_id']; hr=Z['hour']; hi=Z['has_insulin']; hn=Z['has_norepinephrine']; na=Z['n_atoms']
print(pd.Series(np.round(hr[:60],4)).tolist()); print(na[:60].tolist()); print(hi[:60].astype(int).tolist())
# match insulin bundles to rates
trS = tr[tr.stay_id.isin(selS)]
insS = trS[trS.event=='insulin'][['stay_id','hour','rate']]
B = pd.DataFrame({'stay_id':sid,'hour':np.round(hr,4),'hi':hi,'hn':hn,'na':na,'row':np.arange(len(sid))})
insS = insS.assign(hour=np.round(insS.hour,4))
Bi = B[B.hi].merge(insS, on=['stay_id','hour'], how='left')
print('insulin bundles', len(Bi), 'matched rate', Bi.rate.notna().mean())
from sklearn.linear_model import RidgeCV
def probe(Xf, y, groups):
    from sklearn.model_selection import GroupKFold
    pred=np.zeros_like(y)
    for tri,tei in GroupKFold(5).split(Xf,y,groups):
        m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(Xf[tri],y[tri]); pred[tei]=m.predict(Xf[tei])
    return 1-np.mean((y-pred)**2)/np.var(y)
Bi=Bi.dropna(subset=['rate'])
for nm,sl in [('out',slice(0,192)),('h',slice(192,960)),('z',slice(960,992)),('all',slice(0,992))]:
    print(nm,'R2 rate at insulin bundle:', round(probe(Xs[Bi.row.values][:,sl], Bi.rate.values, Bi.stay_id.values),3))
```