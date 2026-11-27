Two findings. Insulin: the best single-constraint decoder has noise of about 1.2 per unit response. Its x1/2 response is 72% of what linearity implies. NE: x2 alone is easy (noise 0.18), but that decoder ignores the x1/2 direction, so NE needs both edit directions constrained. The cross-drug leak is not scored, so I can drop that constraint.

Next: minimum-noise decoders with unit response to both own-drug edits (x2 and x1/2).

```python
def mvdr2(drug,codes,rw,lam):
    G_=[]
    for f_ in [2.0,0.5]:
        e_=EDR[(drug,f_)]; m_=np.isin(cf[e_['gi']],codes)&e_['reached'].astype(bool)
        G_.append(e_['dX'][m_].mean(0)/e_['dd'][m_][:,0].mean())
    G_=np.array(G_).T
    S_=np.cov(Rf[rw].T.astype(np.float64)); S_+=lam*np.trace(S_)/192*np.eye(192)
    SiG=np.linalg.solve(S_,G_); w0=SiG@np.linalg.solve(G_.T@SiG,np.ones(2))
    return w0
VD={}
for drug,codes,rw in [('insulin',[0,1],ins_rows),('norepinephrine',[2,3],map_rows)]:
    for lam in [1e-4,1e-3,1e-2]:
        w0=mvdr2(drug,codes,rw,lam); VD[(drug,lam)]=w0
        print(drug[:3],lam,'noise/unit',round(np.std(Rf[rw]@w0),3),[(k[0][:3]+str(k[1]),resp(w0,k,codes)[0]) for k in keysI+keysN])
```