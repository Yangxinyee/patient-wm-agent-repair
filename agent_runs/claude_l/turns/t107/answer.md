The probe works, and it gives a clean answer.

- **ctx is solved.** With GELU in the gap MLP, my documented-formula ctx matches the model to 1e-6.
- **Both encodings are wrong.** The e and u errors are large (median about 1.5–1.9).

The u error is large even though most bundles have no actions, where I set u = 0. So "absent" is probably not a zero vector: it may be the empty-set encoding or some learned null.

Next I split the e and u errors by bundle type: with or without measurements, with or without actions. I check whether the true u in no-action bundles is one constant vector, and whether it equals the empty-set encoding of `encode_action`. Likewise for e in action-only bundles. For bundles that do contain atoms, I compare several encoder variants: activation, whether `kq` is added, and pooling. Finding exact matches for the empty case and for one encoder variant would complete the input side.

```python
ACTMAP.clear(); ACTMAP.update({'gap':'gelu'})
recs=[]
for s in stays:
    rows=row_of[s]; bl=BUN[s]
    for k in range(len(bl)):
        meas=[a for a in bl[k][1] if not a['act']]; act=[a for a in bl[k][1] if a['act']]
        recs.append((s,k,len(meas),len(act),Xp[rows[k],960+10:960+21],Xp[rows[k],960+21:992]))
noact=np.stack([r[5] for r in recs if r[3]==0]); nomeas=np.stack([r[4] for r in recs if r[2]==0])
print('u at no-action bundles: spread', np.abs(noact-noact[0]).max(), ' value', np.round(noact[0][:4],4))
print('e at no-meas bundles: spread', np.abs(nomeas-nomeas[0]).max(), ' value', np.round(nomeas[0][:4],4))
for ea in ['silu','gelu','relu']:
    ACTMAP.update({'encode.mlp1':ea,'encode.mlp2':ea,'encode_action.mlp1':ea,'encode_action.mlp2':ea})
    ue=set_enc(W0,[],'encode_action').numpy(); ee_=set_enc(W0,[],'encode').numpy()
    print(ea,'empty-enc match: u %.2e  e %.2e'%(np.abs(Pu@ue-noact[0]).max(), np.abs(Pe@ee_-nomeas[0]).max()))
```