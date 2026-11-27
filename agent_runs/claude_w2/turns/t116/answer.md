The decompiled `zerr_custom` differs from my encoder in two ways:
1. END atoms are stripped of their rate and dose (NaN, so has_rate/has_dose = 0 and bins 0) and routed to the measurement encoder.
2. Bundles with no e-atoms get the empty-set e, and bundles with no action get the empty-set u.

I'll add both rules to impl.py as the default encoding and rerun the z-posterior check by bundle kind. Matching `zerr_custom` (about 0.002 for action and end bundles, about 0.013 for finger glucose) settles the encoder. If glucose bundles still differ, the glucose atoms must have another difference, which I would then chase.

```python
IMPL2 = r'''
def encode_bundles(A,W,end_attr=False):
    A=A.copy()
    if not end_attr: A.loc[A.kind==2,['rate','dose']]=np.nan
    feats=atom_features(A); xa=embed(A,feats,W)
    key=A.stay_id.values.astype(np.int64)*100000+A.bidx.values
    ukey,gid=np.unique(key,return_inverse=True)
    nb=len(ukey)
    isact=((A.kind.values==0)&(A.ekind.values==2))
    E,ce=set_encode(xa[torch.as_tensor(~isact)],gid[~isact],nb,W,'encode')
    U,cu=set_encode(xa[torch.as_tensor(isact)],gid[isact],nb,W,'encode_action')
    B=pd.DataFrame({'stay_id':ukey//100000,'bidx':ukey%100000})
    B['minute']=A.groupby(key).minute.first().reindex(ukey).values
    B['t']=A.groupby(key).t.first().reindex(ukey).values
    return B,E,U,(cu>0).numpy(),xa
'''
open('/work/impl.py','a').write(IMPL2); exec(IMPL2)
B0,E0,U0,hasu0,xa0=encode_bundles(A0,WO)
Eo=E0[rowB]; Uo=U0[rowB]
print('zerr new', zerr_with(Eo,Uo))
```