`ins_t` is a boolean flag, not a dose, so that run broke before the unit analysis. `A_` holds dev1 (the lag-1 insulin deviation) but no raw dose. Next I will look for the previous-bundle dose in the row tables `RA_` or `Rcat`, which have the same 27,963 rows. Then I'll finish the analysis: the gate term (half the sum) and readout term (half the difference) of units 26 and 41, tabulated by previous dose on glucose_finger rows that follow an insulin bundle. The test is the same as before: a gate falling off at small doses, or a gate that stays open while the readout is flat.

```python
for nm in ['RA_','Rcat','Dg3']:
    v=globals()[nm]; print(nm, list(v.columns))
pre=np.einsum('drk,uk->dru',WCall.astype(np.float64),W0c[[26,41]])+b0c[[26,41]]
gsum=(pre[...,0]+pre[...,1])/2; gdif=(pre[...,0]-pre[...,1])/2
print('W2 26/41 norms',np.linalg.norm(W2c[:,26]).round(3),np.linalg.norm(W2c[:,41]).round(3),'cos',(W2c[:,26]@W2c[:,41]/np.linalg.norm(W2c[:,26])/np.linalg.norm(W2c[:,41])).round(3))
gf=(A_.code.values==6)
for hv in [False,True]:
    mk_=gf&(A_.hi.values==hv)
    print('hi',hv,'n',mk_.sum(),'gsum mean',gsum[:,mk_].mean().round(2),'q05/q95',np.round(np.quantile(gsum[:,mk_].mean(0),[.05,.5,.95]),2),
          'gdif sd',gdif[:,mk_].mean(0).std().round(3))
```