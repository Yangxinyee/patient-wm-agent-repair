The glucose and MAP "reading directions" in the slot stream are well defined. They are identical across slots, reasonably stable across bundles (cosine 0.64 for glucose, 0.81 for MAP), and roughly reproducible through head 1 alone (fit cosine 0.62 and 0.57). The head-1 value needed is 0.36 per mg/dL of glucose and 1.52 per mmHg of MAP.

Now the innovation write-in directions η (z-token perturbations, shared by all four END patterns). Each η must:
- move head 1's value by the glucose (or MAP) target per unit, with cross-talk to the other reading direction removed;
- leave head-1 and head-3 key alignment unchanged;
- stay orthogonal to the pattern vectors, so LayerNorm stays nearly linear.

I'll then check exactly, for glucose-equivalent shifts up to ±40 mg/dL (and ±20 mmHg), how linear the head-1 value change is and whether the attention takeover margins hold. Linearity within about 10% and margins still beyond ±6 would confirm the design.

```python
DG,DM=DIRS['glucose_finger'],DIRS['map_arterial']; Wo1=Wo[:,48:96].astype(float)
def v_orth(D,Dx):
    M=np.vstack([Wo1, 1e3*(Dx@Wo1)[None]]); return np.linalg.lstsq(M,np.r_[D,0.],rcond=None)[0]
vG=v_orth(DG,DM); vM=v_orth(DM,DG)
for nm,v,D,Dx in [('G',vG,DG,DM),('M',vM,DM,DG)]: print(nm,'fit cos %.3f crosstalk cos %.4f |v| %.3f'%(np.dot(Wo1@v,D)/np.linalg.norm(Wo1@v)/np.linalg.norm(D),np.dot(Wo1@v,Dx)/np.linalg.norm(Wo1@v)/np.linalg.norm(Dx),np.linalg.norm(v)))
CB={}
for p in range(4):
    b=np.where(pid==p)[0]
    CB[p]=[((Qs[b][:,:,h*48:(h+1)*48].reshape(-1,48)@Wk[h*48:(h+1)*48])*g1[None,:]).mean(0)/np.sqrt(48) for h in range(4)]
rp=np.array([np.sqrt((DES[p]**2).mean()) for p in range(4)]); rbar=rp.mean(); print('r_p',np.round(rp,3))
def eta_for(v):
    C=np.vstack([Av, np.ones(192)]+[DES[p] for p in range(4)]+[CB[p][h] for p in range(4) for h in (1,3)])
    d=np.r_[rbar*v, np.zeros(1+4+8)]; return C.T@np.linalg.lstsq(C@C.T,d,rcond=None)[0]
ETA={'G':eta_for(vG),'M':eta_for(vM)}
for nm,eps_list in [('G',[-40,-10,10,40]),('M',[-20,-5,5,20])]:
    eta=ETA[nm]; v=vG if nm=='G' else vM; print(nm,'|eta| per unit %.4f'%np.linalg.norm(eta))
    for p in range(4):
        b=np.where(pid==p)[0]; t0=LNnp(DES[p][None],n1w,n1b)[0]; rows=[]
        for ep in eps_list:
            t=LNnp((DES[p]+ep*eta)[None],n1w,n1b)[0]; dv=Wv[48:96]@(t-t0); lin=np.dot(dv,v)/np.dot(v,v)/ep
            m1=min(((Qs[b][:,:,48:96]@(Wk[48:96]@t))/np.sqrt(48)-Ls[b][:,1,:,:6].max(-1)).min(),99); m3=((Qs[b][:,:,144:192]@(Wk[144:192]@t))/np.sqrt(48)-Ls[b][:,3,:,:6].max(-1)).max()
            rows.append('eps%+d gain %.3f h1m %.1f h3m %.1f'%(ep,lin,m1,m3))
        print(' p',p,' | '.join(rows))
```