With GELU encoders, both empty-set encodings match exactly (about 1e-6). u in no-action bundles is always the encoder's empty-set output, not zero, and the same holds for e at the first action-only bundle. But e varies across action-only bundles (spread 2.3). So something in those bundles reaches the measurement encoder.

The obvious candidate is the infusion END atoms (kind 2). An infusion ending is a recorded world event, not a treatment decision, so it is plausible that END atoms go to the measurement/visit encoder. Only drug START atoms (kind 0) would then be actions.

Next I test this routing with GELU encoders over all bundles. If both e and u errors drop to about 1e-6, the input side of the replica is exact.

```python
ACTMAP.update({'encode.mlp1':'gelu','encode.mlp2':'gelu','encode_action.mlp1':'gelu','encode_action.mlp2':'gelu','gap':'gelu'})
def split_atoms(atoms, route='end_to_e'):
    if route=='end_to_e':
        act=[a for a in atoms if a['act'] and a['kind']==0]; meas=[a for a in atoms if not (a['act'] and a['kind']==0)]
    else:
        act=[a for a in atoms if a['act']]; meas=[a for a in atoms if not a['act']]
    return meas,act
def enc_bundle2(atoms, route='end_to_e'):
    meas,act=split_atoms(atoms,route)
    return set_enc(W0,meas,'encode'), set_enc(W0,act,'encode_action')
for route in ['end_to_e','act_all']:
    ee=[];eu=[];bad=[]
    for s in stays:
        rows=row_of[s]; bl=BUN[s]
        for k in range(len(bl)):
            e,u=enc_bundle2(bl[k][1],route); zt=Xp[rows[k],960:]
            er_e=np.abs(Pe@e.numpy()-zt[10:21]).max(); er_u=np.abs(Pu@u.numpy()-zt[21:32]).max()
            ee.append(er_e); eu.append(er_u)
            if max(er_e,er_u)>1e-4 and len(bad)<5: bad.append((s,k,[(a['code'],a['kind'],a['vb'],a['rb']) for a in bl[k][1]],round(er_e,4),round(er_u,4)))
    ee=np.array(ee); eu=np.array(eu)
    print(route,'e: frac<1e-4 %.4f max %.2e | u: frac<1e-4 %.4f max %.2e'%((ee<1e-4).mean(),ee.max(),(eu<1e-4).mean(),eu.max()))
    for b_ in bad: print('   ',b_)
```