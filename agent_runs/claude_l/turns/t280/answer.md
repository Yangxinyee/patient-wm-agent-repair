The data collection was quick: 320 training stays in 17 s, so the teacher-forced replica is cheap to iterate with. Three findings:
- **The context vector does not encode the expected dose linearly.** Ridge R2 is 0.53 for insulin and 0.37 for norepinephrine, against about 0.87 and 0.94 for the policy GBM. With a linear expected-dose term, much of the policy-predictable dose would leak into the residual. That would cost factual NLL, though not bias the dose-edit readouts.
- **u is easy to copy.** 28–30 PCA components reproduce it with a median relative error of about 0.001.
- **u decodes the dose well** (R2 > 0.99), including for edited doses on average, though the worst insulin case is off by up to 4 U/h.

The architecture file confirms the state update uses the posterior mean, so the z token carries no sampling noise.

Next, I'll test whether a nonlinear decoder that fits in one gelu layer recovers the expected dose much better. It uses random gelu features of the whitened context plus a ridge fit, which is a plain regression and could be copied into z_post's hidden layer. I'll also check:
- which policy inputs the context holds: last glucose, last MAP, previous rates;
- how accurately linear indicators of "drug present" and "insulin/NE present" can be read from u, for gating;
- whether a nonlinear dose-from-u decoder fixes the worst-case error.

If random features reach R2 around 0.8 or more, the residual design stands. If not, the leakage cost must be measured directly in the teacher-forced NLL.

```python
import numpy as np, time
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from scipy.special import erf as serf
t0=time.time()
def gelu(x): return 0.5*x*(1+serf(x/np.sqrt(2)))
GF,MA=CODE_ID['glucose_finger'],CODE_ID['map_arterial']
def last_vals(B_,k):
    v=[None]*4
    for j in range(k-1,-1,-1):
        for a in B_[j][1]:
            if a['kind']!=0: continue
            if a['code']==GF and v[0] is None: v[0]=a['num'][0]
            if a['code']==MA and v[1] is None: v[1]=a['num'][0]
            if a['act'] and a['code']==8 and v[2] is None: v[2]=a['num'][4]
            if a['act'] and a['code']==9 and v[3] is None: v[3]=a['num'][4]
        if None not in v: break
    return [np.nan if x is None else x for x in v]
LV=np.array([last_vals(TRB[d['s']],d['k']) for d in DD])
def cvr2(X,y,alphas=(1e-3,1e-2,1e-1,1,10,100)):
    rc=RidgeCV(alphas=alphas).fit(X,y); p=np.zeros_like(y)
    for a,b in KFold(5,shuffle=True,random_state=0).split(X): p[b]=Ridge(alpha=rc.alpha_).fit(X[a],y[a]).predict(X[b])
    return 1-np.var(y-p)/np.var(y), rc.alpha_
print('decode from ctx (lin R2):',[(nm,round(cvr2(C[~np.isnan(LV[:,j])],LV[~np.isnan(LV[:,j]),j])[0],3)) for j,nm in enumerate(['lastG','lastMAP','prevIns','prevNE'])])
mc=C.mean(0); ev_,EV=np.linalg.eigh(np.cov((C-mc).T)); o=np.argsort(ev_)[::-1]; ev_=ev_[o]; EV=EV[:,o]
kk=60; Wh=EV[:,:kk]/np.sqrt(ev_[:kk]); Cw=(C-mc)@Wh
for nm,y,m in [('ins|pres',ri,hi),('ne|pres',rn,hn),('ins|all',ri,np.ones(len(ri),bool)),('ne|all',rn,np.ones(len(rn),bool))]:
    best=None
    for sc in [0.5,1.0,2.0]:
        g=np.random.default_rng(3); A=g.normal(0,sc/np.sqrt(kk),(kk,140)); cb=g.normal(0,1,140)
        r,al=cvr2(np.hstack([gelu(Cw[m]@A+cb),Cw[m]]),y[m])
        if best is None or r>best[0]: best=(r,sc)
    print(nm,'lin R2',round(cvr2(Cw[m],y[m])[0],3),'RF R2',round(best[0],3),'scale',best[1])
ue=u_empty.numpy().astype(np.float64)
Ud=np.vstack([U,UE]); lab_i=np.r_[hi,TE[:,2]>0]; lab_n=np.r_[hn,TE[:,3]>0]
X1=np.vstack([Ud-ue, np.zeros((1,192))]); 
for nm,lab in [('any',np.ones(len(Ud),bool)),('ins',lab_i),('ne',lab_n)]:
    y=np.r_[lab.astype(float),0.0]; w=np.r_[np.ones(len(Ud)),200.0]
    a=np.linalg.solve((X1*w[:,None]).T@X1+1e-6*np.eye(192),(X1*w[:,None]).T@y); pr=X1@a
    print('indicator',nm,'max err pos/neg/empty',round(np.abs(pr[:-1][lab]-1).max(),4),round(np.abs(pr[:-1][~lab]).max(),4) if (~lab).any() else None,round(abs(pr[-1]),5))
Pu_=np.linalg.svd(Ud-Ud.mean(0),full_matrices=False)[2][:12]; Uc=(Ud-Ud.mean(0))@Pu_.T; Uc=Uc/Uc.std(0)
fac=np.r_[np.ones(len(U)),np.tile([0,.5,2,3,1,1,1,1],len(UE)//8)]; facn=np.r_[np.ones(len(U)),np.tile([1,1,1,1,0,.5,2,3],len(UE)//8)]
for nm,col,lab,fc in [('ins',0,lab_i,fac),('ne',1,lab_n,facn)]:
    y=np.r_[np.where(hi,ri,0) if col==0 else np.where(hn,rn,0), TE[:,col]][lab]
    g=np.random.default_rng(5); A=g.normal(0,1.0/np.sqrt(12),(12,40)); cb=g.normal(0,1,40)
    for tag,Phi in [('lin',Uc[lab]),('rf',np.hstack([gelu(Uc[lab]@A+cb),Uc[lab]]))]:
        p=Ridge(alpha=1e-4).fit(Phi,y).predict(Phi); er=p-y
        print('dose-from-u',nm,tag,'max|err| by factor',[(f,round(np.abs(er[fc[lab]==f]).max(),3)) for f in [0,.5,1,2,3] if (fc[lab]==f).any()])
print('t',round(time.time()-t0))
```