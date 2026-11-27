The notebook is updated. Next I will check whether the insulin dose is linearly readable from E, where the prelude has carried u, and from u itself.

At every insulin bundle of about 40 selS stays, I will compute u_k and E (mean over slots) locally from the model's true previous states. I then fit stay-grouped cross-validated ridge regressions for two targets: the start rate r_t, and its deviation from the policy band mean (my DEV panel).

- **Confirms the plan:** E predicts the rate with high R² (well above the 0.69 found in the head-time state). E should also predict the deviation fairly well, since E mixes the dose (via u) with glucose information (via h).
- **Refutes it:** E's rate R² is no better than the state's. Then the edit must create a clean dose direction in the encoder first.

```python
print('hrs', hrs[:5], hrs[-3:])
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict
rowsE=[]; t0=time.time()
for s in selS[:40]:
    ev_=recs[recs.stay_id==s].sort_values('hour'); B_=build_bundles(ev_,OPTS); rw=np.where(SST==s)[0]
    if len(B_)!=len(rw): continue
    EU_=[enc_eu(al) for _,al in B_]
    for k in range(1,len(B_)):
        ia=[a for a in B_[k][1] if a['code']==CODE['insulin'] and a['kind']==0]
        if not ia: continue
        hp=torch.tensor(SX[rw[k-1],192:960].reshape(4,192),dtype=torch.float64); zk=torch.tensor(SX[rw[k],960:],dtype=torch.float64)
        idx_=list(range(k+1))[-128:]; Mem=torch.stack([EU_[i][0] for i in idx_]); Mage=torch.tensor([(B_[k][0]-B_[i][0])*3600 for i in idx_],dtype=torch.float64)
        y,o,E=core_step_v(hp,(B_[k][0]-B_[k-1][0])*3600,EU_[k][0],EU_[k][1],zk,Mem,Mage,'std')
        hh=int(np.floor(B_[k][0])); hi=int(np.where(hrs==hh)[0][0])
        rowsE.append(dict(s=s,hour=hh,rate=ia[0]['rate'][1],dev=DEV[sidx[s],hi],nne=sum(a['code']==CODE['norepinephrine'] and a['kind']==0 for a in B_[k][1]),
                          E=E.mean(0).numpy(),u=EU_[k][1].numpy(),o=SX[rw[k],:192].astype(np.float64)))
print('n rows',len(rowsE),'time',round(time.time()-t0,1))
gE=np.array([r_['s'] for r_ in rowsE]); XE=np.stack([r_['E'] for r_ in rowsE]); XU=np.stack([r_['u'] for r_ in rowsE]); XO=np.stack([r_['o'] for r_ in rowsE])
yr=np.array([r_['rate'] for r_ in rowsE]); yd=np.array([r_['dev'] for r_ in rowsE])
ok_=np.isfinite(yd); print('dev finite', ok_.sum(), ' dev sd', np.nanstd(yd).round(3), ' rate sd', yr.std().round(3))
def cvr2_(X,y,g):
    p_=cross_val_predict(RidgeCV(alphas=np.logspace(-4,3,15)),X,y,groups=g,cv=GroupKFold(5)); return 1-np.mean((y-p_)**2)/np.var(y)
for nmx,X in [('E',XE),('u',XU),('out',XO)]:
    print(nmx,' R2 rate %.3f   R2 dev %.3f'%(cvr2_(X,yr,gE), cvr2_(X[ok_],yd[ok_],gE[ok_])))
```