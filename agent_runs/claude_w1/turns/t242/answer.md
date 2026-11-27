z sampling dominates. The sampled latent moves the head query by about 3.3 rms, and moves the current decoders' outputs by sd 80 (insulin) and 11 (NE), against a target noise near 1. So in the real model those decoders would be pure noise.

Next I will rebuild the two-constraint minimum-noise decoders with the z-sampling covariance added to the within-cell covariance. I will then report the total noise per unit response for several ridge levels. If the total noise stays near 1.5 or below for insulin and 0.3 for NE, a head edit is still viable. If it is several times larger, I need a different dose signal.

```python
def mvdr2b(drug,codes,rw,lam,zw=1.0):
    G_=[]
    for f_ in [2.0,0.5]:
        e_=EDR[(drug,f_)]; m_=np.isin(cf[e_['gi']],codes)&e_['reached'].astype(bool)
        G_.append(e_['dX'][m_].mean(0)/e_['dd'][m_][:,0].mean())
    G_=np.array(G_).T
    SR=np.cov(Rf[rw].T.astype(np.float64)); SZ=np.cov(DZ[np.isin(CZ,codes)].T)
    S_=SR+zw*SZ; S_+=lam*np.trace(S_)/192*np.eye(192)
    SiG=np.linalg.solve(S_,G_); w0=SiG@np.linalg.solve(G_.T@SiG,np.ones(2))
    return w0, np.sqrt(w0@SR@w0), np.sqrt(w0@SZ@w0)
for drug,codes,rw in [('insulin',[0,1],ins_rows),('norepinephrine',[2,3],map_rows)]:
    for lam in [1e-3,1e-2,1e-1]:
        w0,nR,nZ=mvdr2b(drug,codes,rw,lam)
        print(drug[:3],lam,'|w|',round(np.linalg.norm(w0),1),'noiseR',round(nR,3),'noiseZ',round(nZ,3),'resp',[resp(w0,k,codes)[0] for k in (keysI if drug=='insulin' else keysN)])
```