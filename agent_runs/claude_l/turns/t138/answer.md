Finger-glucose atoms carry no unit (row 0), while lab glucose uses row 5 and MAP row 2. With that, single-finger bundles encode exactly.

The full featurization should now be:
- **finger glucose:** unit 0;
- **lab glucose:** unit 5;
- **arterial and cuff MAP:** unit 2;
- **infusion END atoms:** bare (code, kind, event kind and `kq` only), routed to the observation encoder e;
- **drug start records:** unchanged, routed to the action encoder u.

I'll apply this to every bundle of the 100 stays and check the e and u errors over all 4,932 bundles. If both are about 1e-6 everywhere, the encoders are exact and I can move on to verifying the core step.

```python
UNIT_FIX={6:0,10:5,7:2,5:2}
def fix_atom(a):
    a=dict(a)
    if (not a['act']) and a['ekind']==1: a['unit']=UNIT_FIX[a['code']]
    if a['act'] and a['kind']==2: a['num']=(0,0,0,0,0,0); a['db']=0; a['rb']=0; a['unit']=0; a['rate_unit']=0
    return a
def enc_eu(atoms):
    A=[fix_atom(a) for a in atoms]
    ea=[a for a in A if (not a['act']) or a['kind']==2]; ua=[a for a in A if a['act'] and a['kind']!=2]
    e=set_enc(W0,ea,'encode'); u=set_enc(W0,ua,'encode_action') if len(ua)>0 else torch.zeros(192,dtype=T)
    return e,u
errE=[];errU=[]
for i,r in enumerate(recs):
    e,u=enc_eu(BUN[r[0]][r[1]][1])
    errE.append(np.abs(Pe@e.detach().numpy()-r[4]).max()); errU.append(np.abs(Pu@u.detach().numpy()-r[5]).max())
errE=np.array(errE); errU=np.array(errU)
print('e: median %.2e max %.2e frac>1e-4 %.4f'%(np.median(errE),errE.max(),(errE>1e-4).mean()))
print('u: median %.2e max %.2e frac>1e-4 %.4f'%(np.median(errU),errU.max(),(errU>1e-4).mean()))
bad_i=np.where((errE>1e-4)|(errU>1e-4))[0][:5]
for i in bad_i: print(i, recs[i][:2], [(a['code'],a['kind']) for a in BUN[recs[i][0]][recs[i][1]][1]], errE[i], errU[i])
```