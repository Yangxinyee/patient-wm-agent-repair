The kernel still holds the cf dataset: M has 68,668 rows (doses scaled from h0 = 6/12/16 by m in {0, .5, 1.5, 2, 3}, readings as recorded), plus WC and LG. In the 100 tuning stays the glucose readings are 44% of all readings and the MAP readings are 56%.

Next I fit a linear readout w on with_code for the glucose rows. Under scaled drug-A doses, w·ΔWC should match the tilt each row needs. The needed shift is the true shift minus 0.85 × the previous true shift, minus the model's own direct shift. On factual rows the penalty uses within-group variation, grouping by (code, last bin), because the edit will subtract the group means via embed_transition.

I evaluate out-of-fold by stay, reporting:
- the achieved fraction of the needed shift;
- the exact factual NLL change computed from the stored logits.

**Hypothesis:** the gap needed per step is small, so a gain reaching about 100% of it should cost under about 0.007 NLL per glucose reading. A much larger cost would mean the head-only route is too noisy.

```python
import numpy as np, pandas as pd
Mi=M.reset_index(drop=True); Mi['ri']=np.arange(len(Mi)); GE=['glucose_finger','glucose_lab']
def cenof(ev): c=np.asarray(CEN[ev],float); return np.concatenate([[c[0]],c])
def mix(lg,th,v):
    z=lg+th[:,None,None]*v; z=z-z.max(-1,keepdims=True); p=np.exp(z); return p/p.sum(-1,keepdims=True)
def mstat(lg,th,v,c17):
    p=mix(lg,th,v); m=(p*c17).sum(-1); d=(p*c17*v).sum(-1)-m*(p*v).sum(-1); return m.mean(1),d.mean(1)
def tsolve(lg,v,c17,T,it=10):
    th=np.zeros(len(T)); m0,_=mstat(lg,th,v,c17)
    for _ in range(it):
        m,d=mstat(lg,th,v,c17); th=th+(T-(m-m0))/np.maximum(d,1e-3)
    return th
fac=Mi[(Mi.drug=='none')&Mi.event.isin(GE)].sort_values(['stay','event','hour']).copy()
fac['pobs']=fac.groupby(['stay','event'])['obs'].shift(1)
fac['lb']=[0 if pd.isna(p) else int(fb_(e,p)) for e,p in zip(fac.event,fac.pobs)]
fac['ob']=[int(fb_(e,o)) for e,o in zip(fac.event,fac.obs)]
cf=Mi[(Mi.drug=='insulin')&Mi.event.isin(GE)&(Mi.m!=1.0)].copy()
fk=dict(zip(zip(fac.stay,fac.event,fac.b),fac.ri)); cf['fi']=[fk.get(k,-1) for k in zip(cf.stay,cf.event,cf.b)]
print('n fac',len(fac),'n cf',len(cf),'match',round((cf.fi>=0).mean(),3)); cf=cf[cf.fi>=0]
hs=cf.groupby(['stay','h0','m','hour'])['S'].mean().rename('Sh').reset_index().sort_values(['stay','h0','m','hour'])
hs['Sp']=hs.groupby(['stay','h0','m'])['Sh'].shift(1).fillna(0.0)
cf=cf.merge(hs[['stay','h0','m','hour','Sp']],on=['stay','h0','m','hour'],how='left')
cf['T']=cf.S-0.85*cf.Sp-(cf.mean_cf-cf.mean_f)
c17c=np.stack([cenof(e) for e in cf.event])[:,None,:]; LGc=LG[cf.ri.values]; LGf=LG[fac.ri.values]
th_star=tsolve(LGc,vG,c17c,cf['T'].values)
Xc=WC[cf.ri.values]-WC[cf.fi.values]; Xf=WC[fac.ri.values]
main=((cf.h0==12)&(cf.m==2.0)).values
print('main: n',main.sum(),'S',round(cf.S[main].mean(),2),'model dir',round((cf.mean_cf-cf.mean_f)[main].mean(),2),'T',round(cf['T'][main].mean(),2),'th*',round(th_star[main].mean(),3))
stays=np.array(sorted(fac.stay.unique())); fo_={s:i%5 for i,s in enumerate(stays)}
ff=fac.stay.map(fo_).values; fc=cf.stay.map(fo_).values; gkey=(fac.event+'_'+fac.lb.astype(str)).values
ar=np.arange(len(fac)); nll0=-np.log(mix(LGf,np.zeros(len(fac)),vG).mean(1)[ar,fac.ob.values])
m0c,_=mstat(LGc,np.zeros(len(cf)),vG,c17c); RES={}
for lam in [1,3,10,30,100]:
    thf=np.zeros(len(fac)); thc=np.zeros(len(cf))
    for k in range(5):
        tf=ff!=k; tc=fc!=k
        gm=pd.DataFrame(Xf[tf]).groupby(gkey[tf]).mean(); Xft=Xf-gm.reindex(gkey).fillna(0).values
        A=Xc[tc].T@Xc[tc]+lam*Xft[tf].T@Xft[tf]+1e-3*np.eye(192); w=np.linalg.solve(A,Xc[tc].T@th_star[tc])
        thf[~tf]=Xft[~tf]@w; thc[~tc]=Xc[~tc]@w
    m1,_=mstat(LGc,thc,vG,c17c); ach=m1-m0c; fr=ach[main].mean()/cf['T'][main].mean()
    out=[]
    for g in [1.0,1/max(fr,1e-3)]:
        p=mix(LGf,g*thf,vG).mean(1); out.append(round((-np.log(p[ar,fac.ob.values])-nll0).mean(),4))
    perm=[round(ach[(cf.m==mm).values].mean()/cf['T'][(cf.m==mm).values].mean(),2) for mm in [0,0.5,1.5,2,3]]
    RES[lam]=(thf,thc); print('lam',lam,'frac',round(fr,3),'dNLL g1/gfull',out,'sd thf',round(thf.std(),3),'byM',perm)
```