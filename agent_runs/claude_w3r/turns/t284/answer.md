The norm3 and E spectra are much quieter than I assumed: bottom sd ≈0.0006 and median ≈0.006. So small injections are cleanly readable. With the non-u constraint at 0.1, the rate reads are r and √r, with garbage ≤8 and ≤3.4. Drop the r² read, which leaks 50.

Stage 1, done locally on re-collected bundles, uses five head-2 free value dims (gate, insulin r and √r, norepinephrine r and √r). Each is written into a different bottom-variance direction of the norm3 input. I then fit linear decoders of those values from the perturbed norm3 output and measure:
- the decode noise (target ≤0.06 rate-read units),
- the side effect on E at no-action bundles from the garbage (target ≤0.1).

```python
XA=[];N1T=[];ATT2=[];META=[]
for s in list(dict.fromkeys(SST.tolist()))[:60]:
    Bq,rq,Eq=stay_data(int(s))
    for k in range(len(Bq)):
        att,HO_,tok,n1t=prelude_att(Bq,rq,Eq,k,ret_tokens=True)
        XA.append((tok[:4].double()+torch.cat([HO_[h] for h in range(4)],-1).double()@Wo_t.T).numpy()); N1T.append(n1t.double().numpy()); ATT2.append(att[2].double().numpy()); META.append((s,k))
XA=np.array(XA); N1T=np.array(N1T); ATT2=np.array(ATT2)   # (nb,4,192),(nb,7,192),(nb,4,7)
def dec2(y,lamn=0.1):
    T=np.stack([y,np.sqrt(y)],1); A=Xa.T@Xa/len(Xa)+lamn*Xn1.T@Xn1/len(Xn1)+1e-6*np.eye(193); return np.linalg.solve(A,Xa.T@T/len(Xa))
Wri=dec2(yi); Wrn=dec2(yn)
RW=np.stack([w_g,Wri[:192,0],Wri[:192,1],Wrn[:192,0],Wrn[:192,1]],1)       # 192x5 reads (no intercept)
RC=np.array([0,Wri[192,0],Wri[192,1],Wrn[192,0],Wrn[192,1]])
Vf=np.einsum('bqj,bjd,df->bqf',ATT2,N1T,RW)                                 # value along free dims per slot query
N3a=np.concatenate([LNd(torch.tensor(x),g3t,b3t).numpy() for x in XA]); ev3,U3=np.linalg.eigh(np.cov(N3a.T))
sig0=float(np.median(XA.std(-1))); amp=np.array([.05,.01,.01,.01,.01]); g3n=g3t.numpy()
Dslot=np.stack([amp[f]*sig0*U3[:,1+f]/g3n for f in range(5)],1)            # 192x5 slot write dirs
XA2=XA+np.einsum('bqf,df->bqd',Vf,Dslot); N3b=np.concatenate([LNd(torch.tensor(x),g3t,b3t).numpy() for x in XA2])
Vflat=Vf.reshape(-1,5); N3h=np.hstack([N3b,np.ones((len(N3b),1))]); RHO=np.linalg.lstsq(N3h,Vflat,rcond=None)[0]
print('sig0',round(sig0,2),'decode noise sd per f', np.round((N3h@RHO-Vflat).std(0),4),'V range', np.round(Vflat.min(0),2), np.round(Vflat.max(0),2))
act=np.array([ATT2[b,:,5].mean()>.5 for b in range(len(ATT2))]); print('action bundles',act.sum(),'of',len(act))
EE2=np.concatenate([LNd(torch.tensor(x)+Fnn.gelu(LNd(torch.tensor(x),g3t,b3t)@F1w.T+F1b)@F2w.T+F2b,gpt,bpt).numpy() for x in XA2]); EE1=np.concatenate([LNd(torch.tensor(x)+Fnn.gelu(LNd(torch.tensor(x),g3t,b3t)@F1w.T+F1b)@F2w.T+F2b,gpt,bpt).numpy() for x in XA])
dE=np.linalg.norm(EE2-EE1,axis=1).reshape(len(XA),4).mean(1); print('side-effect |dE| no-action mean/max', round(float(dE[~act].mean()),4), round(float(dE[~act].max()),4),'action mean',round(float(dE[act].mean()),4))
```