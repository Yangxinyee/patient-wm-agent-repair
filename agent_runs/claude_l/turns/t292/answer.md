The u-copy is good with 30 components (max element error 0.025–0.044 against |u| ≈ 12.7). I will center the basis on the reading-bundle u so that this most common case is copied exactly. The linear dose decoder failed, however: max error 5.0 on the grid and 239 on recorded u. Either recorded u differ from my synthetic u for the same rate, or the decoder is badly conditioned or u saturates at high doses.

Next I will check three things:
1. synthetic u against recorded u for the same rates;
2. how much u still changes per dose step at high doses;
3. the column layout of the teacher-forced readings array, which I need for the gain regression.

If the u match and u keeps moving at 3× doses, a nonlinear decoder fitted on dose-augmented data is viable. If u saturates, the residual channel cannot see doses beyond the observed range.

```python
dif=[];cnt=0
for s,B_ in BUN.items():
    for t,atoms in B_:
        ua=[a for a in atoms if a.get('act') and a['kind']==0]
        if not ua: continue
        r,hs_=drug_rates(atoms)
        ud=set_enc(W0,[fix_atom2(a) for a in ua],'encode_action'); us=u_syn(r[8] if hs_[8] else None, r[9] if hs_[9] else None)
        dif.append(float((ud-us).abs().max())); cnt+=1
    if cnt>300: break
print('data vs syn u maxdiff: median %.2e max %.2e n %d'%(np.median(dif),np.max(dif),len(dif)))
for r in [1,3,5,8,12,16,20]:
    print('ins r=%d step.25 |du|=%.3f'%(r,float((u_syn(r,None)-u_syn(r-0.25,None)).norm())),end='; ')
print()
for r in [0.2,0.5,1.0,1.5,2.0,3.0]:
    print('ne r=%.1f step.02 |du|=%.3f'%(r,float((u_syn(None,r)-u_syn(None,r-0.02)).norm())),end='; ')
print('\nRA row0',np.round(TT['RA'][0,:9],3).tolist(),'RA cols',TT['RA'].shape)
```