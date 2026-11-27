The copy and gating parts work well:

- **Copy:** 28 PCA components, centred on the empty-action encoding, reproduce every u to within 0.75% (median 0.12%). This holds for all 15,630 recorded and edited bundles.
- **Gates:** one linear presence score per drug separates present from absent cleanly (insulin ≥ 0.67 vs ≤ 0.04; NE ≥ 0.78 vs ≤ 0.05). The large-offset GELU gate can therefore be made safe.
- **Dose decode:** the linear decode of the rate from u is good in range (RMSE 0.22 for insulin, 0.05 for NE). It breaks at the extreme ×3 doses (insulin errors up to 12 at 25 U/h; NE up to 1.0 at 3.0), because the action encoder's final LayerNorm saturates. The ×3 readouts will be underestimated unless I add a nonlinear decode later.

Next I'll realise the Stage 1 design as actual weights and test it in the exact replica:
- **z_post hidden layer:** 56 units pass through the 28 PCA coordinates of u, using gelu(x) − gelu(−x) = x. 4 units carry the gated residuals, one ±pair per drug; each residual is the dose decoded from u minus a ridge policy decode from the context.
- **z_in:** maps the coordinates back to u, adds κ × residual along each drug's dose tangent ∂u/∂rate, and uses u_empty as the bias.

I'll start with κ = 4 for both drugs, then:
1. check that the realised z token matches the intended one on the training drug bundles;
2. compute the teacher-forced value NLL change on the 100 tuning stays, per code, against the current model.

**Expectation:** z-token error well under 1% of |u − u_empty|. NLL change for insulin about 0, possibly slightly negative; some cost on MAP from the NE part, which tells me how much NE gain the budget allows.

```python
import torch, time
t0=time.time()
def fitlin(X,y,lam=1e-6):
    X1=np.c_[X,np.ones(len(X))]; A_=X1.T@X1+lam*np.eye(X1.shape[1]); return np.linalg.solve(A_,X1.T@y)
k=28; V=Vd[:k]
P={}
for j,c in enumerate(dcodes):
    pres=MU[:,2+j]>0.5; inr=pres&(MU[:,j]<= (12.0 if c==8 else 2.2))
    P[c]=dict(gate=fitlin(U_all,MU[:,2+j]), rate=fitlin(U_all[inr],MU[inr,j],1e-3))
    m=DM[:,4+j]>0.5; rg=Ridge(1.0).fit(DC[m],DM[m,2+j]); P[c]['pi']=(rg.coef_.astype(np.float64),float(rg.intercept_))
    # dose tangent from 60 sample bundles
    tg=[]
    for s in keys[:30]:
        for (t,atoms) in BUN[s]:
            ua=[a for a in atoms if a['act'] and a['kind']==0 and a['code']==c]
            if not ua: continue
            nm=DRUGN[c]; rm,ri=NUMV[(nm,'rate')]['median'],NUMV[(nm,'rate')]['iqr']; r0=ua[0]['num'][4]*ri+rm
            if r0<=0: continue
            up=np.asarray(u_for(atoms,t,(c,1.1),-1.0),dtype=np.float64).ravel(); um=np.asarray(u_for(atoms,t,(c,0.9),-1.0),dtype=np.float64).ravel()
            tg.append((up-um)/(0.2*r0))
    P[c]['tan']=np.mean(tg,0); print(DRUGN[c],'tangent n',len(tg),'norm %.2f'%np.linalg.norm(P[c]['tan']))
def build_W(kap, Bg=60.0, Mg=120.0, base=W0):
    W1=np.zeros((192,576)); b1=np.zeros(192); W2=np.zeros((64,192)); b2=np.zeros(64)
    for j in range(k):
        W1[2*j,384:]=V[j]; b1[2*j]=-V[j]@ue; W1[2*j+1,384:]=-V[j]; b1[2*j+1]=V[j]@ue
        W2[j,2*j]=1; W2[j,2*j+1]=-1
    Zin=np.zeros((192,32)); Zin[:,:k]=V.T
    for jj,c in enumerate(dcodes):
        p=P[c]; g=p['gate']; ra=p['rate']; wpi,bpi=p['pi']
        xu=ra[:-1]; xc=-wpi; x0=ra[-1]-bpi
        gu=Mg*g[:-1]; g0=Bg+Mg*(g[-1]-1)
        for sgn,row in [(1,2*k+2*jj),(-1,2*k+2*jj+1)]:
            W1[row,384:]=sgn*xu+gu; W1[row,:192]=sgn*xc; b1[row]=sgn*x0+g0
        W2[k+jj,2*k+2*jj]=0.5; W2[k+jj,2*k+2*jj+1]=-0.5
        Zin[:,k+jj]=kap[c]*p['tan']
    b2[32:]=base['z_post.2.bias'][32:].numpy()
    Wn=dict(base)
    for n_,a_ in [('z_post.0.weight',W1),('z_post.0.bias',b1),('z_post.2.weight',W2),('z_post.2.bias',b2),('z_in.weight',Zin),('z_in.bias',ue)]:
        Wn[n_]=torch.tensor(a_,dtype=T)
    Wn['z_head.weight']=torch.zeros_like(base['z_head.weight'])
    return Wn
Wk=build_W({8:4.0,9:4.0})
# verify realised z token on DA bundles
sel_=np.random.RandomState(0).choice(len(DC),3000,replace=False)
ctx_=torch.tensor(DC[sel_],dtype=T); u_=torch.tensor(DU[sel_],dtype=T)
z=bmlp(Wk,torch.cat([ctx_,torch.zeros_like(ctx_),u_],-1),'z_post')[:,:32]; zt=blin(Wk,z,'z_in').numpy()
ut=DU[sel_]; cp=ue+(ut-ue)@V.T@V
inj=np.zeros_like(cp)
for jj,c in enumerate(dcodes):
    p=P[c]; pres=DM[sel_,4+jj]>0.5
    r=(np.c_[ut,np.ones(len(ut))]@p['rate'])-(DC[sel_]@p['pi'][0]+p['pi'][1]); inj+=np.where(pres,r,0)[:,None]*4.0*p['tan'][None]
    print(DRUGN[c],'resid sd %.3f'%r[pres].std())
err=np.linalg.norm(zt-(cp+inj),axis=1)/np.linalg.norm(ut-ue,axis=1)
print('realised-vs-intended rel err median %.2e max %.2e ; copy err median %.2e'%(np.median(err),err.max(),np.median(np.linalg.norm(cp-ut,axis=1)/np.linalg.norm(ut-ue,axis=1))))
print('build+verify %.1fs'%(time.time()-t0))
```