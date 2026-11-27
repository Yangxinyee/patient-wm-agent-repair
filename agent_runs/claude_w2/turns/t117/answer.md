Stripping END-atom attributes lowered the action and end-bundle errors, to 0.016 and 0.012. They are still above `zerr_custom`'s 0.002, and finger-glucose bundles have not changed (0.108). So another encoding difference remains, and it acts on glucose bundles.

Next I rebuild `zerr_custom`'s encodings directly with the stored helpers (`atom_feats`, `embed_atoms`, `bundle_enc` on the ATM copy with END attributes removed). I compare them with mine by bundle kind and atom by atom. If the atom vectors match but the bundle encodings differ, the problem is in my pooling or bundle grouping. If the atom vectors already differ for glucose atoms, my feature code differs for glucose.

```python
A2_=ATM.copy(); A2_.loc[A2_.kind==2,['rate','dose']]=np.nan
fe2=atom_features if False else None
num2,vb2,db2,rb2=atom_feats(A2_,'left')
xa2=embed_atoms(A2_,num2,vb2,db2,rb2,umapB,rumapB,True)
isact2=(A2_.ekind.values==2)&(A2_.kind.values!=2)
Ebc,Ubc=bundle_enc(xa2,A2_.bidx.values,isact2,nB,'gelu')
print('global bidx range', A2_.bidx.min(), A2_.bidx.max())
# compare atom vectors (my order vs ATM order)
A0n=A0.copy(); A0n.loc[A0n.kind==2,['rate','dose']]=np.nan
xa0n=embed(A0n,atom_features(A0n),WO)
print('atom diff', float((xa0n-torch.as_tensor(np.asarray(xa2))[torch.as_tensor(ordr)]).abs().max()))
dE=(torch.as_tensor(np.asarray(Ebc)).float()-Eo).abs().mean(1).numpy()
print({k_:round(float(dE[kind_tab[rowB]==k_].mean()),4) for k_ in np.unique(kt)})
# number of e-atoms per bundle compare
cntc=np.bincount(A2_.bidx.values[~isact2],minlength=nB)
key0=A0n.stay_id.values.astype(np.int64)*100000+A0n.bidx.values
uk,gid0=np.unique(key0,return_inverse=True)
isa0=(A0n.kind.values==0)&(A0n.ekind.values==2)
cnt0=np.bincount(gid0[~isa0],minlength=len(uk))[rowB]
print('count mismatch by kind', {k_:int((cntc!=cnt0)[kind_tab[rowB]==k_].sum()) for k_ in np.unique(kt)})
```